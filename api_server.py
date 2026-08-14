import os
import json
import time
import math
import random
import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from src.utils.helpers import haversine_distance, load_config
from src.models.dijkstra_routing import load_congestion_ratios, load_free_flow_speeds, get_predictions_map
from src.features.metrics import compute_fuel_and_emissions
from src.models.osmnx_routing import osmnx_dijkstra_static, osmnx_dijkstra_td, extract_path_segments

app = FastAPI(title="Real-World OSM Routing API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

app_state = {}

def lat_lon_to_grid(lat, lon, bbox, grid_dims):
    min_lat, min_lon, max_lat, max_lon = bbox
    num_rows, num_cols = grid_dims
    row = int((lat - min_lat) / (max_lat - min_lat) * num_rows)
    col = int((lon - min_lon) / (max_lon - min_lon) * num_cols)
    return max(0, min(num_rows - 1, row)), max(0, min(num_cols - 1, col))

@app.on_event("startup")
def startup_event():
    print("Initializing OSMnx Real Road Graph...")
    config = load_config("config/default_config.yaml")
    processed_dir = "data/processed"
    
    # AI Predictions State
    predictions_df = pd.read_csv(os.path.join(processed_dir, "congestion_predictions.csv"))
    global_ratios, cell_ratios = load_congestion_ratios(processed_dir)
    free_flow_speeds = load_free_flow_speeds(processed_dir)
    predictions_map = get_predictions_map(predictions_df)
    
    with open(os.path.join(processed_dir, "synthetic_fleet.json"), 'r') as f:
        fleet_data = json.load(f)
    vehicle_config = fleet_data["fleet"][0]["config"]
    
    bbox = config["preprocessing"]["bbox"]
    grid_dims = (config["spatial_grid"]["num_rows"], config["spatial_grid"]["num_cols"])
    
    # 1. Download or Load OSMnx Graph
    graph_path = os.path.join(processed_dir, "beijing_road_graph.graphml")
    if os.path.exists(graph_path):
        print("Loading cached OSM road graph from disk...")
        G = ox.load_graphml(graph_path)
    else:
        print("Downloading OSM road graph for Central Beijing (15x15km)...")
        # bbox=(left, bottom, right, top) which is (min_lon, min_lat, max_lon, max_lat)
        # 116.32 to 116.48 lon, 39.83 to 39.97 lat
        G = ox.graph_from_bbox(bbox=(116.32, 39.83, 116.48, 39.97), network_type='drive')
        print("Saving graphml to cache...")
        ox.save_graphml(G, graph_path)
        
    print("Extracting largest strongly connected component to ensure routability...")
    import networkx as nx
    largest_cc = max(nx.strongly_connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
        
    print(f"OSM Graph loaded with {len(G.nodes)} nodes and {len(G.edges)} edges.")
    
    # 2. Map OSM Edges to 30x30 Spatial Grid Traffic Predictions
    print("Mapping edges to AI traffic grid...")
    for u, v, k, data in G.edges(keys=True, data=True):
        length = data.get('length', 10.0)
        speed_mps = 8.33 # default 30 km/h
        if 'maxspeed' in data:
            ms = data['maxspeed']
            if isinstance(ms, list): ms = ms[0]
            try: speed_mps = float(ms) * 1000 / 3600
            except: pass
        
        data['static_time'] = length / speed_mps if speed_mps > 0 else length / 8.33
        
        if 'geometry' in data:
            coords = list(data['geometry'].coords)
            mid = coords[len(coords)//2]
            lon_mid, lat_mid = mid[0], mid[1]
        else:
            lon_mid = (G.nodes[u]['x'] + G.nodes[v]['x']) / 2
            lat_mid = (G.nodes[u]['y'] + G.nodes[v]['y']) / 2
            
        row, col = lat_lon_to_grid(lat_mid, lon_mid, bbox, grid_dims)
        data['grid_cell'] = (row, col)
        
    app_state.update({
        "G": G,
        "predictions_map": predictions_map,
        "global_ratios": global_ratios,
        "cell_ratios": cell_ratios,
        "free_flow_speeds": free_flow_speeds,
        "vehicle_config": vehicle_config
    })
    print("Backend ready.")


class Point(BaseModel):
    lat: float
    lon: float

class CompareRequest(BaseModel):
    depot: Point
    deliveries: List[Point]

def evaluate_full_path_metrics(G, path, vehicle_config, is_td=False, start_time_sec=0.0):
    """
    Computes accurate fuel, CO2, and metrics based on the sequence of edges.
    """
    if not path or len(path) < 2:
        return {"distance_km": 0.0, "duration_sec": 0.0, "fuel_l": 0.0, "co2_g": 0.0, "avg_congestion": 0.0}
        
    total_dist = 0.0
    total_dur = 0.0
    speeds, dists, durs, congs = [], [], [], []
    curr_time = start_time_sec
    
    from src.models.osmnx_routing import get_predicted_speed_at_time
    
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        edge_data = G.get_edge_data(u, v)
        if not edge_data: continue
        data = edge_data[min(edge_data.keys())]
        
        dist = data.get('length', 10.0)
        
        if is_td:
            cell = data.get('grid_cell')
            default_speed = dist / data.get('static_time', dist/8.33)
            ff_speed_kmh = app_state["free_flow_speeds"].get(cell, 0.0) if cell else 0.0
            ff_speed_mps = ff_speed_kmh / 3.6 if ff_speed_kmh > 0 else 0.0
            
            speed_mps = get_predicted_speed_at_time(
                cell, curr_time, app_state["predictions_map"], 
                app_state["global_ratios"], app_state["cell_ratios"], 
                ff_speed_mps, default_speed
            )
            dur = dist / speed_mps if speed_mps > 0 else dist / 8.33
            
            # Record congestion
            fractional_hour = (curr_time / 3600.0) % 24
            h1 = int(fractional_hour)
            c1 = app_state["predictions_map"].get((cell[0], cell[1], h1), 0) if cell else 0
            congs.append(c1)
        else:
            dur = data.get('static_time', dist/8.33)
            speed_mps = dist / dur if dur > 0 else 8.33
            congs.append(0)
            
        total_dist += dist
        total_dur += dur
        curr_time += dur
        
        speeds.append(speed_mps * 3.6)
        dists.append(dist)
        durs.append(dur)
        
    fuel_l, co2_g = 0.0, 0.0
    if speeds:
        f, c = compute_fuel_and_emissions(np.array(speeds), np.array(dists), np.array(durs), vehicle_config, 0.0)
        fuel_l = float(np.sum(f))
        co2_g = float(np.sum(c))
        
    return {
        "distance_km": total_dist / 1000.0,
        "duration_sec": total_dur,
        "avg_congestion": float(np.mean(congs)) if congs else 0.0,
        "fuel_l": fuel_l,
        "co2_g": co2_g
    }

def solve_tsp_aco(distance_matrix, num_nodes, num_ants=20, max_iter=50):
    """
    Miniature MMAS ACO solver tailored for the 10-target matrix.
    Returns optimal sequence of indices (starting and ending at 0).
    """
    pheromones = np.ones((num_nodes, num_nodes))
    best_seq = None
    best_cost = float('inf')
    
    alpha = 1.0
    beta = 2.0
    evap = 0.1
    
    # Heuristic is 1/dist
    eta = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and distance_matrix[i][j] > 0:
                eta[i][j] = 1.0 / distance_matrix[i][j]
                
    for _ in range(max_iter):
        all_paths = []
        all_costs = []
        
        for ant in range(num_ants):
            unvisited = set(range(1, num_nodes))
            curr = 0
            path = [0]
            cost = 0.0
            
            while unvisited:
                probs = []
                for nxt in unvisited:
                    p = (pheromones[curr][nxt] ** alpha) * (eta[curr][nxt] ** beta)
                    probs.append((nxt, p))
                    
                total_p = sum(p for _, p in probs)
                if total_p == 0:
                    nxt = random.choice(list(unvisited))
                else:
                    rand = random.uniform(0, total_p)
                    cum = 0
                    for n, p in probs:
                        cum += p
                        if cum >= rand:
                            nxt = n
                            break
                            
                path.append(nxt)
                cost += distance_matrix[curr][nxt]
                unvisited.remove(nxt)
                curr = nxt
                
            path.append(0)
            cost += distance_matrix[curr][0]
            
            all_paths.append(path)
            all_costs.append(cost)
            
            if cost < best_cost:
                best_cost = cost
                best_seq = path
                
        # MMAS Update (only best ant deposits)
        pheromones *= (1.0 - evap)
        
        if best_seq is not None:
            tau_delta = 1.0 / best_cost if best_cost > 0 else 0
            for i in range(len(best_seq) - 1):
                u, v = best_seq[i], best_seq[i+1]
                pheromones[u][v] += tau_delta
            
        # Bounds
        pheromones = np.clip(pheromones, 0.01, 5.0)
        
    return best_seq

def find_nearest_node(G, lon, lat):
    min_dist = float('inf')
    nearest = None
    for n, data in G.nodes(data=True):
        # haversine_distance expects (lat1, lon1, lat2, lon2)
        d = haversine_distance(lat, lon, data['y'], data['x'])
        if d < min_dist:
            min_dist = d
            nearest = n
    return nearest

@app.post("/api/compare")
def compare_routes(req: CompareRequest):
    G = app_state["G"]
    
    # 1. Map requested points to nearest OSM intersections
    input_points = [req.depot] + req.deliveries
    # Check bounds
    for i, p in enumerate(input_points):
        if not (116.32 <= p.lon <= 116.48 and 39.83 <= p.lat <= 39.97):
            raise HTTPException(status_code=400, detail="One or more points are outside the Central Beijing routing area. Please place points closer to the city center.")
            
    target_nodes = []
    for p in input_points:
        node = find_nearest_node(G, p.lon, p.lat)
        target_nodes.append(node)
        
    num_targets = len(target_nodes)
    
    # --- OLD ALGORITHM (Static Dijkstra + Greedy TSP) ---
    t0 = time.time()
    static_matrix = {}
    static_paths = {}
    for i in range(num_targets):
        static_matrix[i] = {}
        static_paths[i] = {}
        for j in range(num_targets):
            if i != j:
                path, dur, dist = osmnx_dijkstra_static(G, target_nodes[i], target_nodes[j])
                static_matrix[i][j] = dur if path else float('inf')
                static_paths[i][j] = path
            else:
                static_matrix[i][j] = 0.0
                
    # Greedy Nearest Neighbor Sequence
    unvisited = set(range(1, num_targets))
    curr = 0
    old_seq = [0]
    while unvisited:
        # Avoid finding distance to disconnected nodes
        valid_targets = [x for x in unvisited if static_matrix[curr][x] != float('inf')]
        if not valid_targets:
            # Graph disconnected, pick random
            nxt = unvisited.pop()
        else:
            nxt = min(valid_targets, key=lambda x: static_matrix[curr][x])
            unvisited.remove(nxt)
        old_seq.append(nxt)
        curr = nxt
    old_seq.append(0)  # Return to depot
    
    # Assemble Old Path
    old_path = []
    for i in range(len(old_seq) - 1):
        u, v = old_seq[i], old_seq[i+1]
        p = static_paths[u].get(v)
        if p:
            old_path.extend(p if i == 0 else p[1:])
            
    old_metrics = evaluate_full_path_metrics(G, old_path, app_state["vehicle_config"], is_td=False)
    old_metrics["runtime_sec"] = time.time() - t0
    
    # --- NEW ALGORITHM (TD-Dijkstra + MMAS ACO) ---
    t1 = time.time()
    departure_time_sec = 8 * 3600.0
    
    td_matrix = {}
    td_paths = {}
    
    for i in range(num_targets):
        td_matrix[i] = {}
        td_paths[i] = {}
        for j in range(num_targets):
            if i != j:
                # We approximate matrix with static departure time for heuristic
                path, dur, dist = osmnx_dijkstra_td(
                    G, target_nodes[i], target_nodes[j], departure_time_sec,
                    app_state["predictions_map"], app_state["global_ratios"],
                    app_state["cell_ratios"], app_state["free_flow_speeds"]
                )
                td_matrix[i][j] = dur if path else float('inf')
            else:
                td_matrix[i][j] = 0.0
                
    # Run MMAS
    new_seq = solve_tsp_aco(td_matrix, num_targets)
    if not new_seq: new_seq = old_seq
    
    # Assemble true Time-Dependent path dynamically
    new_path = []
    curr_time = departure_time_sec
    for i in range(len(new_seq) - 1):
        u_idx, v_idx = new_seq[i], new_seq[i+1]
        p, dur, dist = osmnx_dijkstra_td(
            G, target_nodes[u_idx], target_nodes[v_idx], curr_time,
            app_state["predictions_map"], app_state["global_ratios"],
            app_state["cell_ratios"], app_state["free_flow_speeds"]
        )
        if p:
            new_path.extend(p if i == 0 else p[1:])
            curr_time += dur
            
    new_metrics = evaluate_full_path_metrics(G, new_path, app_state["vehicle_config"], is_td=True, start_time_sec=departure_time_sec)
    new_metrics["runtime_sec"] = time.time() - t1
    
    # 4. Extract Detailed Segments for Frontend Telemetry
    old_segments = extract_path_segments(G, old_path, is_td=False, start_time_sec=departure_time_sec)
    new_segments = extract_path_segments(G, new_path, is_td=True, start_time_sec=departure_time_sec, 
                                         predictions_map=app_state["predictions_map"], 
                                         global_ratios=app_state["global_ratios"], 
                                         cell_ratios=app_state["cell_ratios"], 
                                         free_flow_speeds=app_state["free_flow_speeds"])
    
    return {
        "old_algo": {
            "segments": old_segments,
            "ordered_indices": old_seq,
            "metrics": old_metrics
        },
        "new_algo": {
            "segments": new_segments,
            "ordered_indices": new_seq,
            "metrics": new_metrics
        }
    }
