import heapq
import logging
import os
import numpy as np
import pandas as pd
from src.utils.helpers import haversine_distance
from src.features.metrics import compute_fuel_and_emissions

logger = logging.getLogger("IM-VRM")

def get_cell_centroid(row, col, bbox, grid_dims):
    """
    Compute geodetic latitude and longitude coordinates for a grid cell centroid.
    """
    min_lat, min_lon, max_lat, max_lon = bbox
    num_rows, num_cols = grid_dims
    lat_step = (max_lat - min_lat) / num_rows
    lon_step = (max_lon - min_lon) / num_cols
    lat = min_lat + (row + 0.5) * lat_step
    lon = min_lon + (col + 0.5) * lon_step
    return lat, lon

def build_weighted_adjacency_graph(edges_df, predictions_df, sorted_nodes, config, w1, w2, w3, hour=None):
    """
    Construct a weighted routing graph where each edge's cost is:
      Cost(e) = w1 * norm_duration + w2 * norm_distance + w3 * norm_congestion
    
    Metrics are min-max normalized deterministically across the active graph.
    Returns:
        weighted_adj_dict: dict mapping u_idx -> list of tuples (v_idx, cost, duration_sec, distance_m)
        stats: dict containing normalization min/max values and edge cost distributions
    """
    N = len(sorted_nodes)
    node_to_idx = {node: idx for idx, node in enumerate(sorted_nodes)}
    
    # 1. Parse bounding box and grid dimensions
    bbox = config["preprocessing"]["bbox"]
    grid_dims = (config["spatial_grid"]["num_rows"], config["spatial_grid"]["num_cols"])
    
    # 2. Map congestion predictions to cells
    # If hour is provided, filter for that hour. Otherwise, compute average per grid cell.
    congestion_map = {}
    if not predictions_df.empty:
        if hour is not None:
            df_hr = predictions_df[predictions_df["hour"] == int(hour)]
            for _, row in df_hr.iterrows():
                congestion_map[(int(row["grid_row"]), int(row["grid_col"]))] = float(row["predicted_congestion_level"])
        else:
            # Average across all test hours
            df_avg = predictions_df.groupby(["grid_row", "grid_col"])["predicted_congestion_level"].mean().reset_index()
            for _, row in df_avg.iterrows():
                congestion_map[(int(row["grid_row"]), int(row["grid_col"]))] = float(row["predicted_congestion_level"])
                
    # 3. Gather raw edge components to prepare for normalization
    raw_edges = []
    for _, edge in edges_df.iterrows():
        u = (int(edge["grid_row"]), int(edge["grid_col"]))
        v = (int(edge["next_row"]), int(edge["next_col"]))
        
        if u in node_to_idx and v in node_to_idx:
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            
            # Duration (sec)
            dur = float(edge["avg_duration_sec"])
            
            # Distance (meters)
            lat_u, lon_u = get_cell_centroid(u[0], u[1], bbox, grid_dims)
            lat_v, lon_v = get_cell_centroid(v[0], v[1], bbox, grid_dims)
            dist = haversine_distance(lat_u, lon_u, lat_v, lon_v)
            
            # Destination Cell Congestion Level
            cong = congestion_map.get(v, 0.0) # default to FreeFlow
            
            raw_edges.append((u_idx, v_idx, dur, dist, cong))
            
    if not raw_edges:
        logger.warning("No valid edges found to build weighted adjacency graph.")
        return {}, {}
        
    # Extract components for deterministic normalization bounds
    durations = [e[2] for e in raw_edges]
    distances = [e[3] for e in raw_edges]
    congestions = [e[4] for e in raw_edges]
    
    min_dur, max_dur = min(durations), max(durations)
    min_dist, max_dist = min(distances), max(distances)
    min_cong, max_cong = min(congestions), max(congestions)
    
    logger.info(f"Deterministic Normalization Bounds:")
    logger.info(f"  Duration min/max: {min_dur:.2f} s / {max_dur:.2f} s")
    logger.info(f"  Distance min/max: {min_dist:.2f} m / {max_dist:.2f} m")
    logger.info(f"  Congestion min/max: {min_cong:.2f} / {max_cong:.2f}")
    
    # 4. Construct normalized edge costs
    weighted_adj_dict = {}
    costs = []
    
    for u_idx, v_idx, dur, dist, cong in raw_edges:
        # Standard Min-Max normalization mapping to [0, 1]
        norm_dur = (dur - min_dur) / (max_dur - min_dur) if max_dur > min_dur else 0.0
        norm_dist = (dist - min_dist) / (max_dist - min_dist) if max_dist > min_dist else 0.0
        norm_cong = (cong - min_cong) / (max_cong - min_cong) if max_cong > min_cong else cong / 2.0
        
        # Weighted cost scoring
        cost = w1 * norm_dur + w2 * norm_dist + w3 * norm_cong
        costs.append(cost)
        
        if u_idx not in weighted_adj_dict:
            weighted_adj_dict[u_idx] = []
        weighted_adj_dict[u_idx].append((v_idx, cost, dur, dist))
        
    cost_stats = {
        "mean": float(np.mean(costs)),
        "std": float(np.std(costs)),
        "min": float(np.min(costs)),
        "max": float(np.max(costs)),
        "p25": float(np.percentile(costs, 25)),
        "p50": float(np.percentile(costs, 50)),
        "p75": float(np.percentile(costs, 75))
    }
    
    logger.info(f"Weighted Edge Cost Statistics: mean={cost_stats['mean']:.4f}, std={cost_stats['std']:.4f}")
    
    normalization_bounds = {
        "duration": {"min": min_dur, "max": max_dur},
        "distance": {"min": min_dist, "max": max_dist},
        "congestion": {"min": min_cong, "max": max_cong}
    }
    
    stats = {
        "normalization_bounds": normalization_bounds,
        "cost_distribution": cost_stats
    }
    
    return weighted_adj_dict, stats

def dijkstra_weighted_pathfinder(start_idx, end_idx, weighted_adj_dict, N):
    """
    Weighted Dijkstra pathfinder to minimize cumulative Cost(e) while tracking duration and distance.
    """
    if start_idx == end_idx:
        return [start_idx], 0.0, 0.0, 0.0
        
    costs = {i: float('inf') for i in range(N)}
    durations = {i: float('inf') for i in range(N)}
    distances = {i: float('inf') for i in range(N)}
    parents = {i: None for i in range(N)}
    
    costs[start_idx] = 0.0
    durations[start_idx] = 0.0
    distances[start_idx] = 0.0
    
    # Priority Queue stores tuples: (cumulative_cost, node_idx, duration, distance)
    pq = [(0.0, start_idx, 0.0, 0.0)]
    
    while pq:
        curr_cost, curr_node, curr_dur, curr_dist = heapq.heappop(pq)
        
        if curr_node == end_idx:
            break
            
        if curr_cost > costs[curr_node]:
            continue
            
        for neighbor, edge_cost, duration, distance in weighted_adj_dict.get(curr_node, []):
            new_cost = curr_cost + edge_cost
            new_dur = curr_dur + duration
            new_dist = curr_dist + distance
            
            if new_cost < costs[neighbor]:
                costs[neighbor] = new_cost
                durations[neighbor] = new_dur
                distances[neighbor] = new_dist
                parents[neighbor] = curr_node
                heapq.heappush(pq, (new_cost, neighbor, new_dur, new_dist))
                
    if costs[end_idx] == float('inf'):
        return None, float('inf'), float('inf'), float('inf')
        
    # Reconstruct path
    path = []
    curr = end_idx
    while curr is not None:
        path.append(curr)
        curr = parents[curr]
    path.reverse()
    
    return path, costs[end_idx], durations[end_idx], distances[end_idx]

# Module-level caches to avoid rebuilding transition and congestion lookups inside loops
_edge_stats_cache = None
_congestion_map_cache = None
_cached_edges_id = None
_cached_predictions_id = None

def evaluate_path_metrics(path, edges_df, sorted_nodes, predictions_df, vehicle_config):
    """
    Compute actual metrics for a route path including travel distance, travel duration,
    average congestion, fuel consumption, and CO2 emissions.
    """
    global _edge_stats_cache, _congestion_map_cache, _cached_edges_id, _cached_predictions_id

    if not path or len(path) < 2:
        return {
            "distance_km": 0.0,
            "duration_sec": 0.0,
            "avg_congestion": 0.0,
            "fuel_l": 0.0,
            "co2_g": 0.0
        }
        
    node_to_idx = {node: idx for idx, node in enumerate(sorted_nodes)}
    
    # 1. Check or build edge_stats cache
    edges_id = id(edges_df)
    if _edge_stats_cache is None or _cached_edges_id != edges_id:
        _edge_stats_cache = {}
        for _, edge in edges_df.iterrows():
            u = (int(edge["grid_row"]), int(edge["grid_col"]))
            v = (int(edge["next_row"]), int(edge["next_col"]))
            if u in node_to_idx and v in node_to_idx:
                _edge_stats_cache[(node_to_idx[u], node_to_idx[v])] = float(edge["avg_duration_sec"])
        _cached_edges_id = edges_id
        
    edge_stats = _edge_stats_cache
    
    # 2. Check or build congestion_map cache
    pred_id = id(predictions_df)
    if _congestion_map_cache is None or _cached_predictions_id != pred_id:
        _congestion_map_cache = {}
        if not predictions_df.empty:
            df_avg = predictions_df.groupby(["grid_row", "grid_col"])["predicted_congestion_level"].mean().reset_index()
            for _, row in df_avg.iterrows():
                _congestion_map_cache[(int(row["grid_row"]), int(row["grid_col"]))] = float(row["predicted_congestion_level"])
        _cached_predictions_id = pred_id
        
    congestion_map = _congestion_map_cache
            
    # Calculate cell centroids for distance
    bbox = [39.5, 116.0, 40.3, 116.8] # Beijing default, override if configuration has it
    grid_dims = (30, 30)
    
    total_dist_m = 0.0
    total_dur_s = 0.0
    total_fuel_l = 0.0
    total_co2_g = 0.0
    
    congestions_visited = []
    speeds = []
    dists = []
    durs = []
    
    # Process each transition segment
    for i in range(len(path) - 1):
        u_idx = path[i]
        v_idx = path[i+1]
        
        u_cell = sorted_nodes[u_idx]
        v_cell = sorted_nodes[v_idx]
        
        congestions_visited.append(congestion_map.get(v_cell, 0.0))
        
        dur = edge_stats.get((u_idx, v_idx), 0.0)
        
        lat_u, lon_u = get_cell_centroid(u_cell[0], u_cell[1], bbox, grid_dims)
        lat_v, lon_v = get_cell_centroid(v_cell[0], v_cell[1], bbox, grid_dims)
        dist = haversine_distance(lat_u, lon_u, lat_v, lon_v)
        
        total_dist_m += dist
        total_dur_s += dur
        
        speed = (dist / dur) * 3.6 if dur > 0 else 0.0
        speeds.append(speed)
        dists.append(dist)
        durs.append(dur)
        
    # Average congestion (always visited at least starting cell node)
    if not congestions_visited:
        congestions_visited = [congestion_map.get(sorted_nodes[path[0]], 0.0)]
    mean_cong = float(np.mean(congestions_visited))
    
    # Vectorized fuel/emissions computation across all steps in the path
    if speeds:
        fuel_step, co2_step = compute_fuel_and_emissions(
            np.array(speeds), 
            np.array(dists), 
            np.array(durs), 
            vehicle_config, 
            payload_kg=0.0
        )
        total_fuel_l = float(np.sum(fuel_step))
        total_co2_g = float(np.sum(co2_step))
        
    return {
        "distance_km": total_dist_m / 1000.0,
        "duration_sec": total_dur_s,
        "avg_congestion": mean_cong,
        "fuel_l": total_fuel_l,
        "co2_g": total_co2_g
    }

def optimize_greedy_routes(baseline_routes_data, weighted_adj_dict, edges_df, sorted_nodes, predictions_df, fleet_data):
    """
    Re-plan routes between visited delivery nodes using weighted Dijkstra.
    Preserves the visit sequence from the Greedy stage strictly.
    """
    N = len(sorted_nodes)
    node_to_idx = {node: idx for idx, node in enumerate(sorted_nodes)}
    
    # Map fleet vehicle config lookup
    fleet_configs = {v["vehicle_id"]: v["config"] for v in fleet_data["fleet"]}
    
    optimized_report = {}
    comparison_metrics = []
    
    for v_id, route_data in baseline_routes_data.items():
        # Reconstruct the target sequence visited by this vehicle
        # Sequence: [depot, target1, target2, ..., targetK, depot]
        target_sequence = []
        depot_cell = tuple(route_data["mapped_depot_cell"])
        if depot_cell in node_to_idx:
            target_sequence.append(node_to_idx[depot_cell])
            
        for step in route_data.get("decision_trace", []):
            if step.get("action") == "return_to_depot":
                if step.get("path_found"):
                    path_nodes = step.get("path_nodes", [])
                    if path_nodes:
                        target_sequence.append(path_nodes[-1])
            else:
                if step.get("path_found") and step.get("selected_target") is not None:
                    target_sequence.append(step["selected_target"])
                    
        # If vehicle could not even start (target_sequence is empty or has length 1), skip re-planning
        if len(target_sequence) < 2:
            logger.warning(f"Vehicle {v_id} has invalid target sequence. Skipping.")
            continue
            
        # Re-plan transitions using weighted Dijkstra pathfinder
        optimized_route_nodes = [target_sequence[0]]
        total_dijkstra_cost = 0.0
        path_valid = True
        
        for i in range(len(target_sequence) - 1):
            u = target_sequence[i]
            v = target_sequence[i+1]
            
            path, cost, dur, dist = dijkstra_weighted_pathfinder(u, v, weighted_adj_dict, N)
            
            if path is None:
                path_valid = False
                logger.warning(f"Dijkstra could not find path between {sorted_nodes[u]} and {sorted_nodes[v]} for vehicle {v_id}.")
                break
                
            optimized_route_nodes.extend(path[1:])
            total_dijkstra_cost += cost
            
        if not path_valid:
            continue
            
        vehicle_config = fleet_configs.get(v_id)
        
        # Calculate green metrics for both Greedy baseline and Dijkstra-optimized routes
        greedy_nodes = route_data["route_cells"]
        # Map greedy route cells back to node indices
        greedy_path_nodes = [node_to_idx[tuple(c)] for c in greedy_nodes if tuple(c) in node_to_idx]
        
        greedy_metrics = evaluate_path_metrics(greedy_path_nodes, edges_df, sorted_nodes, predictions_df, vehicle_config)
        dijkstra_metrics = evaluate_path_metrics(optimized_route_nodes, edges_df, sorted_nodes, predictions_df, vehicle_config)
        
        # Absolute and relative improvements
        dist_diff = greedy_metrics["distance_km"] - dijkstra_metrics["distance_km"]
        dur_diff = greedy_metrics["duration_sec"] - dijkstra_metrics["duration_sec"]
        cong_diff = greedy_metrics["avg_congestion"] - dijkstra_metrics["avg_congestion"]
        fuel_diff = greedy_metrics["fuel_l"] - dijkstra_metrics["fuel_l"]
        co2_diff = greedy_metrics["co2_g"] - dijkstra_metrics["co2_g"]
        
        def pct_red(diff, base):
            return (diff / base * 100) if base > 0 else 0.0
            
        metrics_entry = {
            "vehicle_id": v_id,
            "vehicle_type": route_data["type"],
            "depot_id": route_data["depot_id"],
            
            # Greedy Baseline
            "greedy_distance_km": greedy_metrics["distance_km"],
            "greedy_duration_sec": greedy_metrics["duration_sec"],
            "greedy_avg_congestion": greedy_metrics["avg_congestion"],
            "greedy_fuel_l": greedy_metrics["fuel_l"],
            "greedy_co2_g": greedy_metrics["co2_g"],
            
            # Dijkstra Optimized
            "dijkstra_distance_km": dijkstra_metrics["distance_km"],
            "dijkstra_duration_sec": dijkstra_metrics["duration_sec"],
            "dijkstra_avg_congestion": dijkstra_metrics["avg_congestion"],
            "dijkstra_fuel_l": dijkstra_metrics["fuel_l"],
            "dijkstra_co2_g": dijkstra_metrics["co2_g"],
            
            # Improvement Absolute values
            "improvement_distance_km": dist_diff,
            "improvement_duration_sec": dur_diff,
            "improvement_congestion": cong_diff,
            "improvement_fuel_l": fuel_diff,
            "improvement_co2_g": co2_diff,
            
            # Improvement Percentages (reduction %)
            "reduction_distance_pct": pct_red(dist_diff, greedy_metrics["distance_km"]),
            "reduction_duration_pct": pct_red(dur_diff, greedy_metrics["duration_sec"]),
            "reduction_congestion_pct": pct_red(cong_diff, greedy_metrics["avg_congestion"]),
            "reduction_fuel_pct": pct_red(fuel_diff, greedy_metrics["fuel_l"]),
            "reduction_co2_pct": pct_red(co2_diff, greedy_metrics["co2_g"])

        }
        
        comparison_metrics.append(metrics_entry)
        
        optimized_report[v_id] = {
            "vehicle_id": v_id,
            "type": route_data["type"],
            "depot_id": route_data["depot_id"],
            "depot_cell": route_data["depot_cell"],
            "mapped_depot_cell": route_data["mapped_depot_cell"],
            "delivery_cells": route_data["delivery_cells"],
            "route_cells": [[int(sorted_nodes[idx][0]), int(sorted_nodes[idx][1])] for idx in optimized_route_nodes],
            "dijkstra_cost": float(total_dijkstra_cost),
            
            # Performance metrics
            "distance_km": dijkstra_metrics["distance_km"],
            "duration_sec": dijkstra_metrics["duration_sec"],
            "avg_congestion": dijkstra_metrics["avg_congestion"],
            "fuel_l": dijkstra_metrics["fuel_l"],
            "co2_g": dijkstra_metrics["co2_g"]
        }
        
    return optimized_report, comparison_metrics

# ==============================================================================
# PHASE 4B: TIME-DEPENDENT CONTEXT-AWARE DIJKSTRA ROUTING IMPLEMENTATION
# ==============================================================================

_global_ratios = None
_cell_ratios = None
_free_flow_speeds = None
_predictions_map_cache = None
_predictions_map_id = None

def load_congestion_ratios(processed_dir):
    """
    Load grid_congestion_stats.csv and dynamically derive average speed ratios
    per congestion level, both globally and cell-specifically.
    """
    global _global_ratios, _cell_ratios
    if _global_ratios is not None and _cell_ratios is not None:
        return _global_ratios, _cell_ratios

    stats_path = os.path.join(processed_dir, "grid_congestion_stats.csv")
    if not os.path.exists(stats_path):
        logger.warning(f"grid_congestion_stats.csv not found at {stats_path}. Using hardcoded fallback ratios.")
        _global_ratios = {0: 0.85, 1: 0.55, 2: 0.25}
        _cell_ratios = {}
        return _global_ratios, _cell_ratios

    try:
        df = pd.read_csv(stats_path)
        # Calculate global average speed ratios per congestion level
        global_ratios_df = df.groupby("congestion_level")["speed_ratio"].mean()
        _global_ratios = {0: 0.85, 1: 0.55, 2: 0.25}
        for lvl, val in global_ratios_df.items():
            _global_ratios[int(lvl)] = float(val)

        # Calculate cell-specific average speed ratios per congestion level
        # Only keep cells with at least 3 historical measurements to avoid noise
        cell_grp = df.groupby(["grid_row", "grid_col", "congestion_level"]).agg(
            mean_ratio=("speed_ratio", "mean"),
            sum_count=("count", "sum")
        ).reset_index()

        _cell_ratios = {}
        for _, row in cell_grp.iterrows():
            if row["sum_count"] >= 3:
                key = (int(row["grid_row"]), int(row["grid_col"]), int(row["congestion_level"]))
                _cell_ratios[key] = float(row["mean_ratio"])

        logger.info(f"Derived data-driven congestion speed ratios. Global: {_global_ratios}. Cell-specific entries: {len(_cell_ratios)}")
    except Exception as e:
        logger.error(f"Error deriving congestion speed ratios: {str(e)}. Using fallback ratios.")
        _global_ratios = {0: 0.85, 1: 0.55, 2: 0.25}
        _cell_ratios = {}

    return _global_ratios, _cell_ratios

def load_free_flow_speeds(processed_dir):
    """
    Load cell_free_flow_speeds.csv mapping cell coordinates to free-flow speeds.
    """
    global _free_flow_speeds
    if _free_flow_speeds is not None:
        return _free_flow_speeds

    ff_path = os.path.join(processed_dir, "cell_free_flow_speeds.csv")
    if not os.path.exists(ff_path):
        logger.warning(f"cell_free_flow_speeds.csv not found at {ff_path}.")
        _free_flow_speeds = {}
        return _free_flow_speeds

    try:
        df = pd.read_csv(ff_path)
        _free_flow_speeds = {}
        for _, row in df.iterrows():
            _free_flow_speeds[(int(row["grid_row"]), int(row["grid_col"]))] = float(row["free_flow_speed"])
        logger.info(f"Loaded {len(_free_flow_speeds)} cell free-flow speeds.")
    except Exception as e:
        logger.error(f"Error loading free-flow speeds: {str(e)}.")
        _free_flow_speeds = {}

    return _free_flow_speeds

def get_predictions_map(predictions_df):
    """
    Construct a fast lookup mapping for XGBoost congestion forecasts:
    (row, col, hour) -> predicted_congestion_level
    """
    global _predictions_map_cache, _predictions_map_id
    pred_id = id(predictions_df)
    if _predictions_map_cache is not None and _predictions_map_id == pred_id:
        return _predictions_map_cache

    _predictions_map_cache = {}
    if not predictions_df.empty:
        for _, row in predictions_df.iterrows():
            key = (int(row["grid_row"]), int(row["grid_col"]), int(row["hour"]))
            _predictions_map_cache[key] = int(row["predicted_congestion_level"])
    _predictions_map_id = pred_id
    return _predictions_map_cache

def get_predicted_speed_at_time(cell, t, free_flow_speeds, predictions_map, global_ratios, cell_ratios, default_speed_mps):
    """
    Compute continuous, FIFO-compliant speed (in m/s) at fractional departure time t.
    Supports both traditional dictionary-based lookups and fast list-of-lists indexing.
    """
    if isinstance(predictions_map, list):
        # cell is node_idx in fast list mode
        node_idx = cell
        ff_speed_mps = free_flow_speeds[node_idx]
        if ff_speed_mps <= 0.0:
            return default_speed_mps

        fractional_hour = (t / 3600.0) % 24
        h1 = int(fractional_hour)
        h2 = (h1 + 1) % 24
        lambda_weight = fractional_hour - h1

        c1 = predictions_map[node_idx][h1]
        c2 = predictions_map[node_idx][h2]

        ratio1 = cell_ratios[node_idx][c1]
        ratio2 = cell_ratios[node_idx][c2]

        v1 = ff_speed_mps * ratio1
        v2 = ff_speed_mps * ratio2

        v_interp = (1.0 - lambda_weight) * v1 + lambda_weight * v2
        return max(v_interp, 0.556)

    ff_speed_kmh = free_flow_speeds.get(cell, None)
    if ff_speed_kmh is None or ff_speed_kmh < 5.0:
        return default_speed_mps

    ff_speed_mps = ff_speed_kmh / 3.6

    # Determine adjacent hours and linear interpolation weight
    fractional_hour = (t / 3600.0) % 24
    h1 = int(fractional_hour)
    h2 = (h1 + 1) % 24
    lambda_weight = fractional_hour - h1

    # Look up predicted levels
    c1 = predictions_map.get((cell[0], cell[1], h1), 0)
    c2 = predictions_map.get((cell[0], cell[1], h2), 0)

    # Resolve ratios (cell-specific with global fallback)
    ratio1 = cell_ratios.get((cell[0], cell[1], c1), global_ratios.get(c1, 0.85))
    ratio2 = cell_ratios.get((cell[0], cell[1], c2), global_ratios.get(c2, 0.85))

    v1 = ff_speed_mps * ratio1
    v2 = ff_speed_mps * ratio2

    # Linearly interpolate to guarantee FIFO compliance
    v_interp = (1.0 - lambda_weight) * v1 + lambda_weight * v2

    # Enforce safe minimum threshold (2.0 km/h = 0.556 m/s)
    return max(v_interp, 0.556)

def dijkstra_time_dependent_pathfinder(
    start_idx,
    end_idx,
    departure_time_sec,
    weighted_adj_dict,
    N,
    predictions_map,
    sorted_nodes,
    free_flow_speeds,
    global_ratios,
    cell_ratios,
    normalization_bounds,
    w1, w2, w3
):
    """
    Time-dependent Dijkstra pathfinder to minimize cumulative edge cost on a dynamic network.
    """
    if start_idx == end_idx:
        return [start_idx], 0.0, 0.0, 0.0

    min_dur = normalization_bounds["duration"]["min"]
    max_dur = normalization_bounds["duration"]["max"]
    min_dist = normalization_bounds["distance"]["min"]
    max_dist = normalization_bounds["distance"]["max"]
    min_cong = normalization_bounds["congestion"]["min"]
    max_cong = normalization_bounds["congestion"]["max"]

    costs = {i: float('inf') for i in range(N)}
    durations = {i: float('inf') for i in range(N)}
    distances = {i: float('inf') for i in range(N)}
    parents = {i: None for i in range(N)}

    costs[start_idx] = 0.0
    durations[start_idx] = 0.0
    distances[start_idx] = 0.0

    pq = [(0.0, start_idx, float(departure_time_sec), 0.0)]
    
    is_list_mode = isinstance(predictions_map, list)

    while pq:
        curr_cost, curr_node, curr_time, curr_dist = heapq.heappop(pq)

        if curr_node == end_idx:
            break

        if curr_cost > costs[curr_node]:
            continue

        for neighbor, _, base_dur, distance in weighted_adj_dict.get(curr_node, []):
            # 1. Traversal speed at departure time
            default_speed_mps = distance / base_dur if base_dur > 0 else 5.0
            
            if is_list_mode:
                speed_mps = get_predicted_speed_at_time(
                    neighbor,
                    curr_time,
                    free_flow_speeds,
                    predictions_map,
                    global_ratios,
                    cell_ratios,
                    default_speed_mps
                )
            else:
                neighbor_cell = sorted_nodes[neighbor]
                speed_mps = get_predicted_speed_at_time(
                    neighbor_cell,
                    curr_time,
                    free_flow_speeds,
                    predictions_map,
                    global_ratios,
                    cell_ratios,
                    default_speed_mps
                )

            # 2. Dynamic Traversal Duration
            travel_time_sec = distance / speed_mps

            # 3. Dynamic Congestion Level
            fractional_hour = (curr_time / 3600.0) % 24
            h1 = int(fractional_hour)
            h2 = (h1 + 1) % 24
            lambda_weight = fractional_hour - h1
            
            if is_list_mode:
                c1 = predictions_map[neighbor][h1]
                c2 = predictions_map[neighbor][h2]
            else:
                neighbor_cell = sorted_nodes[neighbor]
                c1 = predictions_map.get((neighbor_cell[0], neighbor_cell[1], h1), 0)
                c2 = predictions_map.get((neighbor_cell[0], neighbor_cell[1], h2), 0)
            
            congestion_level = (1.0 - lambda_weight) * c1 + lambda_weight * c2

            # 4. Multi-Objective Edge Cost
            norm_dur = (travel_time_sec - min_dur) / (max_dur - min_dur) if max_dur > min_dur else 0.0
            norm_dist = (distance - min_dist) / (max_dist - min_dist) if max_dist > min_dist else 0.0
            norm_cong = (congestion_level - min_cong) / (max_cong - min_cong) if max_cong > min_cong else congestion_level / 2.0

            # Clamp normalized terms to [0, 1]
            norm_dur = max(0.0, min(1.0, norm_dur))
            norm_dist = max(0.0, min(1.0, norm_dist))
            norm_cong = max(0.0, min(1.0, norm_cong))

            edge_cost = w1 * norm_dur + w2 * norm_dist + w3 * norm_cong

            new_cost = curr_cost + edge_cost
            new_time = curr_time + travel_time_sec
            new_dist = curr_dist + distance

            if new_cost < costs[neighbor]:
                costs[neighbor] = new_cost
                durations[neighbor] = new_time - departure_time_sec
                distances[neighbor] = new_dist
                parents[neighbor] = curr_node
                heapq.heappush(pq, (new_cost, neighbor, new_time, new_dist))

    if costs[end_idx] == float('inf'):
        return None, float('inf'), float('inf'), float('inf')

    path = []
    curr = end_idx
    while curr is not None:
        path.append(curr)
        curr = parents[curr]
    path.reverse()

    return path, costs[end_idx], durations[end_idx], distances[end_idx]

def evaluate_path_metrics_td(
    path, 
    edges_df, 
    sorted_nodes, 
    predictions_map, 
    free_flow_speeds, 
    global_ratios, 
    cell_ratios, 
    vehicle_config, 
    start_time_sec
):
    """
    Time-dependent evaluation of actual path metrics (emissions, fuel, travel duration, distance).
    """
    if not path or len(path) < 2:
        return {
            "distance_km": 0.0,
            "duration_sec": 0.0,
            "avg_congestion": 0.0,
            "fuel_l": 0.0,
            "co2_g": 0.0
        }

    # Setup default bbox and grid_dims
    bbox = [39.5, 116.0, 40.3, 116.8]
    grid_dims = (30, 30)

    node_to_idx = {node: idx for idx, node in enumerate(sorted_nodes)}
    edge_stats = {}
    for _, edge in edges_df.iterrows():
        u = (int(edge["grid_row"]), int(edge["grid_col"]))
        v = (int(edge["next_row"]), int(edge["next_col"]))
        if u in node_to_idx and v in node_to_idx:
            edge_stats[(node_to_idx[u], node_to_idx[v])] = float(edge["avg_duration_sec"])

    curr_time = float(start_time_sec)
    total_dist_m = 0.0
    congestions_visited = []
    speeds = []
    dists = []
    durs = []
    
    is_list_mode = isinstance(predictions_map, list)

    for i in range(len(path) - 1):
        u_idx = path[i]
        v_idx = path[i+1]
        u_cell = sorted_nodes[u_idx]
        v_cell = sorted_nodes[v_idx]

        lat_u, lon_u = get_cell_centroid(u_cell[0], u_cell[1], bbox, grid_dims)
        lat_v, lon_v = get_cell_centroid(v_cell[0], v_cell[1], bbox, grid_dims)
        dist = haversine_distance(lat_u, lon_u, lat_v, lon_v)

        base_dur = edge_stats.get((u_idx, v_idx), 0.0)
        default_speed_mps = dist / base_dur if base_dur > 0 else 5.0

        if is_list_mode:
            speed_mps = get_predicted_speed_at_time(
                v_idx,
                curr_time,
                free_flow_speeds,
                predictions_map,
                global_ratios,
                cell_ratios,
                default_speed_mps
            )
        else:
            speed_mps = get_predicted_speed_at_time(
                v_cell,
                curr_time,
                free_flow_speeds,
                predictions_map,
                global_ratios,
                cell_ratios,
                default_speed_mps
            )

        travel_time_sec = dist / speed_mps

        fractional_hour = (curr_time / 3600.0) % 24
        h1 = int(fractional_hour)
        h2 = (h1 + 1) % 24
        lambda_weight = fractional_hour - h1
        
        if is_list_mode:
            c1 = predictions_map[v_idx][h1]
            c2 = predictions_map[v_idx][h2]
        else:
            c1 = predictions_map.get((v_cell[0], v_cell[1], h1), 0)
            c2 = predictions_map.get((v_cell[0], v_cell[1], h2), 0)
            
        congestion_val = (1.0 - lambda_weight) * c1 + lambda_weight * c2
        congestions_visited.append(congestion_val)

        total_dist_m += dist
        speeds.append(speed_mps * 3.6)
        dists.append(dist)
        durs.append(travel_time_sec)

        curr_time += travel_time_sec

    if not congestions_visited:
        h_idx = int((start_time_sec / 3600.0) % 24)
        if is_list_mode:
            congestions_visited = [predictions_map[path[0]][h_idx]]
        else:
            congestions_visited = [predictions_map.get((sorted_nodes[path[0]][0], sorted_nodes[path[0]][1], h_idx), 0)]
            
    mean_cong = float(np.mean(congestions_visited))

    total_dur_s = curr_time - start_time_sec
    total_fuel_l = 0.0
    total_co2_g = 0.0

    if speeds:
        fuel_step, co2_step = compute_fuel_and_emissions(
            np.array(speeds),
            np.array(dists),
            np.array(durs),
            vehicle_config,
            payload_kg=0.0
        )
        total_fuel_l = float(np.sum(fuel_step))
        total_co2_g = float(np.sum(co2_step))

    return {
        "distance_km": total_dist_m / 1000.0,
        "duration_sec": total_dur_s,
        "avg_congestion": mean_cong,
        "fuel_l": total_fuel_l,
        "co2_g": total_co2_g
    }

def optimize_routes_time_dependent(
    baseline_routes_data,
    weighted_adj_dict,
    edges_df,
    sorted_nodes,
    predictions_df,
    fleet_data,
    start_time_sec,
    free_flow_speeds,
    global_ratios,
    cell_ratios,
    normalization_bounds,
    w1, w2, w3
):
    """
    Re-plan routes between visited delivery nodes using Time-Dependent Dijkstra.
    Preserves the visit sequence from the baseline stage strictly.
    Supports on-the-fly profiling comparison and verification.
    """
    N = len(sorted_nodes)
    node_to_idx = {node: idx for idx, node in enumerate(sorted_nodes)}
    predictions_map = get_predictions_map(predictions_df)
    fleet_configs = {v["vehicle_id"]: v["config"] for v in fleet_data["fleet"]}

    # Pre-map cell data to fast node-indexed lists
    ff_speed_mps_list = [0.0] * N
    for idx, cell in enumerate(sorted_nodes):
        ff_speed_kmh = free_flow_speeds.get(cell, 0.0)
        if ff_speed_kmh >= 5.0:
            ff_speed_mps_list[idx] = ff_speed_kmh / 3.6

    predictions_grid = [[0] * 24 for _ in range(N)]
    for idx, cell in enumerate(sorted_nodes):
        for h in range(24):
            predictions_grid[idx][h] = predictions_map.get((cell[0], cell[1], h), 0)

    cell_ratios_grid = [[0.0] * 3 for _ in range(N)]
    for idx, cell in enumerate(sorted_nodes):
        for lvl in range(3):
            cell_ratios_grid[idx][lvl] = cell_ratios.get((cell[0], cell[1], lvl), global_ratios.get(lvl, 0.85))

    # Run baseline profiling comparison for correctness and performance stats if vehicle count <= 50
    run_comparison = len(baseline_routes_data) <= 50
    
    if run_comparison:
        import time
        logger.info("Running baseline Time-Dependent Dijkstra for profiling and equivalence verification...")
        start_baseline_time = time.time()
        
        baseline_report = {}
        for v_id, route_data in baseline_routes_data.items():
            target_sequence = []
            depot_cell = tuple(route_data["mapped_depot_cell"])
            if depot_cell in node_to_idx:
                target_sequence.append(node_to_idx[depot_cell])
            for step in route_data.get("decision_trace", []):
                if step.get("action") == "return_to_depot":
                    if step.get("path_found"):
                        path_nodes = step.get("path_nodes", [])
                        if path_nodes:
                            target_sequence.append(path_nodes[-1])
                else:
                    if step.get("path_found") and step.get("selected_target") is not None:
                        target_sequence.append(step["selected_target"])
            if len(target_sequence) < 2:
                continue
                
            optimized_route_nodes = [target_sequence[0]]
            total_cost = 0.0
            path_valid = True
            curr_time = float(start_time_sec)
            for i in range(len(target_sequence) - 1):
                u = target_sequence[i]
                v = target_sequence[i+1]
                path, cost, dur, dist = dijkstra_time_dependent_pathfinder(
                    u, v, curr_time, weighted_adj_dict, N,
                    predictions_map, sorted_nodes, free_flow_speeds,
                    global_ratios, cell_ratios, normalization_bounds,
                    w1, w2, w3
                )
                if path is None:
                    path_valid = False
                    break
                optimized_route_nodes.extend(path[1:])
                total_cost += cost
                curr_time += dur
            if not path_valid:
                continue
                
            vehicle_config = fleet_configs.get(v_id)
            td_metrics = evaluate_path_metrics_td(
                optimized_route_nodes, edges_df, sorted_nodes, predictions_map,
                free_flow_speeds, global_ratios, cell_ratios, vehicle_config, start_time_sec
            )
            baseline_report[v_id] = {
                "route_cells": [[int(sorted_nodes[idx][0]), int(sorted_nodes[idx][1])] for idx in optimized_route_nodes],
                "dijkstra_cost": float(total_cost),
                "distance_km": td_metrics["distance_km"],
                "duration_sec": td_metrics["duration_sec"],
                "avg_congestion": td_metrics["avg_congestion"],
                "fuel_l": td_metrics["fuel_l"],
                "co2_g": td_metrics["co2_g"]
            }
        
        baseline_duration = time.time() - start_baseline_time
        logger.info(f"Baseline Time-Dependent Dijkstra completed in {baseline_duration:.4f} seconds.")

    import time
    logger.info("Running optimized Time-Dependent Dijkstra...")
    start_opt_time = time.time()
    
    optimized_report = {}
    comparison_metrics = []
    
    for v_id, route_data in baseline_routes_data.items():
        target_sequence = []
        depot_cell = tuple(route_data["mapped_depot_cell"])
        if depot_cell in node_to_idx:
            target_sequence.append(node_to_idx[depot_cell])
            
        for step in route_data.get("decision_trace", []):
            if step.get("action") == "return_to_depot":
                if step.get("path_found"):
                    path_nodes = step.get("path_nodes", [])
                    if path_nodes:
                        target_sequence.append(path_nodes[-1])
            else:
                if step.get("path_found") and step.get("selected_target") is not None:
                    target_sequence.append(step["selected_target"])
                    
        if len(target_sequence) < 2:
            continue
            
        optimized_route_nodes = [target_sequence[0]]
        total_cost = 0.0
        path_valid = True
        curr_time = float(start_time_sec)
        
        for i in range(len(target_sequence) - 1):
            u = target_sequence[i]
            v = target_sequence[i+1]
            
            path, cost, dur, dist = dijkstra_time_dependent_pathfinder(
                u, v, curr_time, weighted_adj_dict, N,
                predictions_grid, sorted_nodes, ff_speed_mps_list,
                global_ratios, cell_ratios_grid, normalization_bounds,
                w1, w2, w3
            )
            
            if path is None:
                path_valid = False
                break
                
            optimized_route_nodes.extend(path[1:])
            total_cost += cost
            curr_time += dur
            
        if not path_valid:
            continue
            
        vehicle_config = fleet_configs.get(v_id)
        
        td_metrics = evaluate_path_metrics_td(
            optimized_route_nodes, edges_df, sorted_nodes, predictions_grid,
            ff_speed_mps_list, global_ratios, cell_ratios_grid, vehicle_config, start_time_sec
        )
        
        greedy_nodes = route_data["route_cells"]
        greedy_path_nodes = [node_to_idx[tuple(c)] for c in greedy_nodes if tuple(c) in node_to_idx]
        
        greedy_metrics_at_start = evaluate_path_metrics_td(
            greedy_path_nodes, edges_df, sorted_nodes, predictions_grid,
            ff_speed_mps_list, global_ratios, cell_ratios_grid, vehicle_config, start_time_sec
        )
        
        dist_diff = greedy_metrics_at_start["distance_km"] - td_metrics["distance_km"]
        dur_diff = greedy_metrics_at_start["duration_sec"] - td_metrics["duration_sec"]
        cong_diff = greedy_metrics_at_start["avg_congestion"] - td_metrics["avg_congestion"]
        fuel_diff = greedy_metrics_at_start["fuel_l"] - td_metrics["fuel_l"]
        co2_diff = greedy_metrics_at_start["co2_g"] - td_metrics["co2_g"]
        
        def pct_red(diff, base):
            return (diff / base * 100) if base > 0 else 0.0
            
        metrics_entry = {
            "vehicle_id": v_id,
            "vehicle_type": route_data["type"],
            "depot_id": route_data["depot_id"],
            
            # Greedy Baseline under Dynamic Traffic
            "greedy_distance_km": greedy_metrics_at_start["distance_km"],
            "greedy_duration_sec": greedy_metrics_at_start["duration_sec"],
            "greedy_avg_congestion": greedy_metrics_at_start["avg_congestion"],
            "greedy_fuel_l": greedy_metrics_at_start["fuel_l"],
            "greedy_co2_g": greedy_metrics_at_start["co2_g"],
            
            # TD-Dijkstra Optimized
            "dijkstra_distance_km": td_metrics["distance_km"],
            "dijkstra_duration_sec": td_metrics["duration_sec"],
            "dijkstra_avg_congestion": td_metrics["avg_congestion"],
            "dijkstra_fuel_l": td_metrics["fuel_l"],
            "dijkstra_co2_g": td_metrics["co2_g"],
            
            # Improvements
            "improvement_distance_km": dist_diff,
            "improvement_duration_sec": dur_diff,
            "improvement_congestion": cong_diff,
            "improvement_fuel_l": fuel_diff,
            "improvement_co2_g": co2_diff,
            
            "reduction_distance_pct": pct_red(dist_diff, greedy_metrics_at_start["distance_km"]),
            "reduction_duration_pct": pct_red(dur_diff, greedy_metrics_at_start["duration_sec"]),
            "reduction_congestion_pct": pct_red(cong_diff, greedy_metrics_at_start["avg_congestion"]),
            "reduction_fuel_pct": pct_red(fuel_diff, greedy_metrics_at_start["fuel_l"]),
            "reduction_co2_pct": pct_red(co2_diff, greedy_metrics_at_start["co2_g"])
        }
        
        comparison_metrics.append(metrics_entry)
        
        optimized_report[v_id] = {
            "vehicle_id": v_id,
            "type": route_data["type"],
            "depot_id": route_data["depot_id"],
            "depot_cell": route_data["depot_cell"],
            "mapped_depot_cell": route_data["mapped_depot_cell"],
            "delivery_cells": route_data["delivery_cells"],
            "route_cells": [[int(sorted_nodes[idx][0]), int(sorted_nodes[idx][1])] for idx in optimized_route_nodes],
            "dijkstra_cost": float(total_cost),
            "distance_km": td_metrics["distance_km"],
            "duration_sec": td_metrics["duration_sec"],
            "avg_congestion": td_metrics["avg_congestion"],
            "fuel_l": td_metrics["fuel_l"],
            "co2_g": td_metrics["co2_g"]
        }
        
    opt_duration = time.time() - start_opt_time
    logger.info(f"Optimized Time-Dependent Dijkstra completed in {opt_duration:.4f} seconds.")

    if run_comparison:
        # Equivalence checks
        logger.info("Performing route and metric equivalence check between baseline and optimized runs...")
        for v_id, opt_route in optimized_report.items():
            assert v_id in baseline_report, f"Vehicle {v_id} missing in baseline report!"
            base_route = baseline_report[v_id]
            
            # Verify route cells are exactly identical
            assert opt_route["route_cells"] == base_route["route_cells"], \
                f"Route mismatch for vehicle {v_id}!"
                
            # Verify metrics are identical
            assert abs(opt_route["distance_km"] - base_route["distance_km"]) < 1e-5, \
                f"Distance mismatch for vehicle {v_id}!"
            assert abs(opt_route["duration_sec"] - base_route["duration_sec"]) < 1e-3, \
                f"Duration mismatch for vehicle {v_id}!"
            assert abs(opt_route["fuel_l"] - base_route["fuel_l"]) < 1e-5, \
                f"Fuel mismatch for vehicle {v_id}!"
            assert abs(opt_route["co2_g"] - base_route["co2_g"]) < 1e-3, \
                f"CO2 mismatch for vehicle {v_id}!"
                
        logger.info("[+] Equivalence check passed! Routes and metrics are 100% identical.")
        
        # Save comparison metrics to JSON
        speedup = baseline_duration / opt_duration if opt_duration > 0 else 1.0
        num_vehicles = len(baseline_routes_data)
        
        runtime_stats = {
            "num_vehicles": num_vehicles,
            "baseline_total_runtime_sec": baseline_duration,
            "optimized_total_runtime_sec": opt_duration,
            "speedup_factor": speedup,
            "avg_baseline_runtime_per_vehicle_sec": baseline_duration / num_vehicles if num_vehicles > 0 else 0.0,
            "avg_optimized_runtime_per_vehicle_sec": opt_duration / num_vehicles if num_vehicles > 0 else 0.0
        }
        
        comparison_json_path = "outputs/runtime_comparison.json"
        with open(comparison_json_path, 'w') as f:
            json.dump(runtime_stats, f, indent=2)
        logger.info(f"Saved runtime comparison profiling to: {comparison_json_path}")

    return optimized_report, comparison_metrics
