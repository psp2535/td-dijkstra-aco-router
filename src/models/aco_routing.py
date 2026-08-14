import logging
import json
import numpy as np
import pandas as pd
import torch
from src.utils.helpers import haversine_distance
from src.features.metrics import compute_fuel_and_emissions
from src.models.greedy_routing import dijkstra_pathfinder

logger = logging.getLogger("IM-VRM")

def get_cell_centroid(row, col, bbox, grid_dims):
    """
    Compute geodetic coordinates for a grid cell centroid.
    """
    min_lat, min_lon, max_lat, max_lon = bbox
    num_rows, num_cols = grid_dims
    lat_step = (max_lat - min_lat) / num_rows
    lon_step = (max_lon - min_lon) / num_cols
    lat = min_lat + (row + 0.5) * lat_step
    lon = min_lon + (col + 0.5) * lon_step
    return lat, lon

def evaluate_route_metrics_fast(path, edge_stats, congestion_map, sorted_nodes, vehicle_config, bbox, grid_dims):
    """
    Fast path metrics evaluation using pre-computed lookups.
    """
    if not path or len(path) < 2:
        return {
            "distance_km": 0.0,
            "duration_sec": 0.0,
            "avg_congestion": 0.0,
            "fuel_l": 0.0,
            "co2_g": 0.0
        }
        
    total_dist_m = 0.0
    total_dur_s = 0.0
    total_fuel_l = 0.0
    total_co2_g = 0.0
    congestions_visited = []
    
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
        
        fuel_step, co2_step = compute_fuel_and_emissions(
            np.array([speed]), 
            np.array([dist]), 
            np.array([dur]), 
            vehicle_config, 
            payload_kg=0.0
        )
        
        total_fuel_l += float(fuel_step[0])
        total_co2_g += float(co2_step[0])
        
    if not congestions_visited:
        congestions_visited = [congestion_map.get(sorted_nodes[path[0]], 0.0)]
    mean_cong = float(np.mean(congestions_visited))
    
    return {
        "distance_km": total_dist_m / 1000.0,
        "duration_sec": total_dur_s,
        "avg_congestion": mean_cong,
        "fuel_l": total_fuel_l,
        "co2_g": total_co2_g
    }

def aco_route_planner(depot_node_idx, delivery_node_indices, embeddings, adj_dict, N, 
                      predictions_df, sorted_nodes, edges_df, vehicle_config, greedy_baseline, aco_config=None):
    """
    Runs Ant Colony Optimization (ACO) to optimize delivery target sequence selection for a vehicle.
    ACO replaces the greedy target sequence search.
    """
    if aco_config is None:
        aco_config = {}
        
    # Hyperparameters
    num_ants = aco_config.get("ants", 15)
    max_iterations = aco_config.get("iterations", 25)
    alpha = aco_config.get("alpha", 1.0)           # pheromone weight
    beta = aco_config.get("beta", 2.0)             # heuristic weight
    rho_global = aco_config.get("rho_global", 0.2) # global pheromone decay
    rho_local = aco_config.get("rho_local", 0.1)   # local pheromone decay during traversal
    q0 = aco_config.get("q0", 0.7)                 # exploration vs exploitation balance
    Q = aco_config.get("Q", 1.0)                   # global reinforcement scaling constant
    
    # Heuristic scaling coefficients
    h_alpha1 = aco_config.get("alpha1", 1.0)       # embedding distance coef
    h_alpha2 = aco_config.get("alpha2", 1.5)       # congestion coef
    h_alpha3 = aco_config.get("alpha3", 1.0)       # duration coef (scaled in mins)
    h_alpha4 = aco_config.get("alpha4", 1.0)       # distance coef (scaled in km)
    epsilon = 1e-6
    
    # Objective weights for evaluating paths
    w_dist = aco_config.get("w_distance", 0.2)
    w_dur = aco_config.get("w_duration", 0.2)
    w_cong = aco_config.get("w_congestion", 0.2)
    w_fuel = aco_config.get("w_fuel", 0.2)
    w_co2 = aco_config.get("w_co2", 0.2)
    w_penalty = aco_config.get("w_penalty", 100.0) # Heavy penalty for unvisited nodes
    
    # 1. Candidate nodes mapping
    aco_nodes = [depot_node_idx] + list(delivery_node_indices)
    K = len(aco_nodes)
    node_to_pos = {node_idx: pos for pos, node_idx in enumerate(aco_nodes)}
    
    # 2. Build lookups for fast static data lookup
    node_to_idx = {node: idx for idx, node in enumerate(sorted_nodes)}
    
    edge_stats = {}
    for _, edge in edges_df.iterrows():
        u = (int(edge["grid_row"]), int(edge["grid_col"]))
        v = (int(edge["next_row"]), int(edge["next_col"]))
        if u in node_to_idx and v in node_to_idx:
            edge_stats[(node_to_idx[u], node_to_idx[v])] = float(edge["avg_duration_sec"])
            
    bbox = [39.5, 116.0, 40.3, 116.8] # Beijing default coordinates
    grid_dims = (30, 30)
    
    congestion_map = {}
    if not predictions_df.empty:
        df_avg = predictions_df.groupby(["grid_row", "grid_col"])["predicted_congestion_level"].mean().reset_index()
        for _, row in df_avg.iterrows():
            congestion_map[(int(row["grid_row"]), int(row["grid_col"]))] = float(row["predicted_congestion_level"])
            
    # 3. Precompute and Cache pairwise transition costs and metrics
    cost_cache = {}
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            u_idx = aco_nodes[i]
            v_idx = aco_nodes[j]
            
            # Dijkstra path search on grid graph
            path, dur, dist_m = dijkstra_pathfinder(u_idx, v_idx, adj_dict, N)
            
            if path is not None:
                # Compute green metrics for this segment transition
                metrics = evaluate_route_metrics_fast(
                    path, edge_stats, congestion_map, sorted_nodes, vehicle_config, bbox, grid_dims
                )
                cost_cache[(u_idx, v_idx)] = {
                    "path": path,
                    "duration_sec": dur,
                    "distance_m": dist_m,
                    "congestion": metrics["avg_congestion"],
                    "fuel_l": metrics["fuel_l"],
                    "co2_g": metrics["co2_g"]
                }
            else:
                cost_cache[(u_idx, v_idx)] = None
                
    # 4. Initialize pheromone matrix Tau
    Tau = np.ones((K, K), dtype=float)
    Tau_min, Tau_max = 0.05, 5.0
    
    # Reconstruct GNN embeddings array for candidates
    embeddings_np = embeddings.numpy()
    
    best_overall_route = None
    best_overall_objective = float('inf')
    best_overall_metrics = {}
    best_overall_target_seq = []
    
    convergence_history = []
    pheromone_matrix_history = []
    
    # Seed generator deterministically per vehicle for reproducible sampling
    rng = np.random.default_rng(seed=42)
    
    # 5. Main ACO iterations loop
    for iteration in range(max_iterations):
        ant_routes = []
        ant_objectives = []
        ant_metrics_list = []
        ant_target_seqs = []
        
        for ant in range(num_ants):
            curr_node = depot_node_idx
            curr_pos = node_to_pos[curr_node]
            
            unvisited = set(delivery_node_indices)
            target_seq = [curr_node]
            full_path = [curr_node]
            
            unvisited_penalty = 0
            
            # Track metrics accumulation directly
            accum_dist_m = 0.0
            accum_dur_s = 0.0
            accum_fuel_l = 0.0
            accum_co2_g = 0.0
            congestions_visited = []
            
            # Sub-loop to incrementally visit all targets
            while unvisited:
                candidates = list(unvisited)
                valid_candidates = []
                heuristic_vals = []
                prob_weights = []
                
                for cand in candidates:
                    cache = cost_cache.get((curr_node, cand))
                    if cache is None:
                        continue # Unreachable
                        
                    # L2 distance in GNN embedding space
                    d_embed = float(np.linalg.norm(embeddings_np[cand] - embeddings_np[curr_node], ord=2))
                    
                    # Cost metrics scaled to similar ranges
                    cong = cache["congestion"]
                    dur_min = cache["duration_sec"] / 60.0
                    dist_km = cache["distance_m"] / 1000.0
                    
                    # Heuristic function value (inverse sum of scaled parameters)
                    eta = 1.0 / (h_alpha1 * d_embed + h_alpha2 * cong + h_alpha3 * dur_min + h_alpha4 * dist_km + epsilon)
                    
                    cand_pos = node_to_pos[cand]
                    tau = Tau[curr_pos, cand_pos]
                    
                    # Weight combining pheromone strength and heuristic score
                    weight = (tau ** alpha) * (eta ** beta)
                    
                    valid_candidates.append(cand)
                    heuristic_vals.append(eta)
                    prob_weights.append(weight)
                    
                if not valid_candidates:
                    # Ant is stuck (cannot reach any remaining target nodes). End route construction.
                    unvisited_penalty = len(unvisited)
                    break
                    
                # Selection logic (Pseudo-random proportional rule)
                prob_weights = np.array(prob_weights)
                best_cand_idx = np.argmax(prob_weights)
                
                selected_node = None
                r = rng.uniform(0.0, 1.0)
                
                if r < q0:
                    # Deterministic exploitation
                    selected_node = valid_candidates[best_cand_idx]
                else:
                    # Probabilistic exploration
                    total_w = np.sum(prob_weights)
                    if total_w > 0:
                        probs = prob_weights / total_w
                        selected_node = rng.choice(valid_candidates, p=probs)
                    else:
                        selected_node = rng.choice(valid_candidates)
                        
                selected_pos = node_to_pos[selected_node]
                
                # Local Pheromone Update on selected transition (directed)
                Tau[curr_pos, selected_pos] = (1.0 - rho_local) * Tau[curr_pos, selected_pos] + rho_local * 1.0
                # Enforce bounds
                Tau[curr_pos, selected_pos] = np.clip(Tau[curr_pos, selected_pos], Tau_min, Tau_max)
                
                # Append segment path and accumulate metrics from cache
                cache = cost_cache[(curr_node, selected_node)]
                full_path.extend(cache["path"][1:])
                target_seq.append(selected_node)
                
                accum_dist_m += cache["distance_m"]
                accum_dur_s += cache["duration_sec"]
                accum_fuel_l += cache["fuel_l"]
                accum_co2_g += cache["co2_g"]
                congestions_visited.append(cache["congestion"])
                
                unvisited.remove(selected_node)
                curr_node = selected_node
                curr_pos = selected_pos
                
            # Return to depot segment
            if curr_node != depot_node_idx:
                cache = cost_cache.get((curr_node, depot_node_idx))
                if cache is not None:
                    full_path.extend(cache["path"][1:])
                    target_seq.append(depot_node_idx)
                    
                    accum_dist_m += cache["distance_m"]
                    accum_dur_s += cache["duration_sec"]
                    accum_fuel_l += cache["fuel_l"]
                    accum_co2_g += cache["co2_g"]
                    congestions_visited.append(cache["congestion"])
                else:
                    unvisited_penalty += 1
                    
            # Compute physical metrics (summed from cached values)
            if not congestions_visited:
                congestions_visited = [congestion_map.get(sorted_nodes[full_path[0]], 0.0)]
            mean_cong = float(np.mean(congestions_visited))
            
            metrics = {
                "distance_km": accum_dist_m / 1000.0,
                "duration_sec": accum_dur_s,
                "avg_congestion": mean_cong,
                "fuel_l": accum_fuel_l,
                "co2_g": accum_co2_g
            }
            
            # Normalize objective components relative to frozen Greedy baseline
            g_dist = greedy_baseline.get("distance_km", 1.0)
            g_dur = greedy_baseline.get("duration_sec", 1.0)
            g_cong = greedy_baseline.get("avg_congestion", 1.0)
            g_fuel = greedy_baseline.get("fuel_l", 1.0)
            g_co2 = greedy_baseline.get("co2_g", 1.0)
            
            norm_dist = metrics["distance_km"] / g_dist if g_dist > 0 else metrics["distance_km"]
            norm_dur = metrics["duration_sec"] / g_dur if g_dur > 0 else metrics["duration_sec"]
            norm_cong = metrics["avg_congestion"] / g_cong if g_cong > 0 else metrics["avg_congestion"]
            norm_fuel = metrics["fuel_l"] / g_fuel if g_fuel > 0 else metrics["fuel_l"]
            norm_co2 = metrics["co2_g"] / g_co2 if g_co2 > 0 else metrics["co2_g"]
            
            # Multi-objective cost value
            objective = (w_dist * norm_dist + w_dur * norm_dur + w_cong * norm_cong +
                         w_fuel * norm_fuel + w_co2 * norm_co2 + w_penalty * unvisited_penalty)
            
            ant_routes.append(full_path)
            ant_objectives.append(objective)
            ant_metrics_list.append(metrics)
            ant_target_seqs.append(target_seq)
            
        if not ant_objectives:
            continue
            
        # Get best ant in this iteration
        best_idx_iter = np.argmin(ant_objectives)
        best_obj_iter = ant_objectives[best_idx_iter]
        best_route_iter = ant_routes[best_idx_iter]
        best_metrics_iter = ant_metrics_list[best_idx_iter]
        best_target_seq_iter = ant_target_seqs[best_idx_iter]
        
        # 6. Global Pheromone Update (Evaporation + Best-Ant Reinforcement)
        Tau = (1.0 - rho_global) * Tau
        
        # Reinforce edges used by the iteration-best route
        for i in range(len(best_target_seq_iter) - 1):
            u_pos = node_to_pos[best_target_seq_iter[i]]
            v_pos = node_to_pos[best_target_seq_iter[i+1]]
            Tau[u_pos, v_pos] += Q / best_obj_iter
            
        # Enforce max-min pheromone bounds
        Tau = np.clip(Tau, Tau_min, Tau_max)
        
        # Log convergence values
        convergence_history.append({
            "iteration": int(iteration),
            "best_objective": float(best_obj_iter),
            "mean_objective": float(np.mean(ant_objectives)),
            "best_distance_km": float(best_metrics_iter["distance_km"]),
            "best_congestion": float(best_metrics_iter["avg_congestion"])
        })
        
        # Log pheromone matrix snapshot
        pheromone_matrix_history.append(Tau.copy().tolist())
        
        # Update global best route
        if best_obj_iter < best_overall_objective:
            best_overall_objective = best_obj_iter
            best_overall_route = best_route_iter
            best_overall_metrics = best_metrics_iter
            best_overall_target_seq = best_target_seq_iter
            
    # Re-evaluate final route outputs mapping centroids
    route_cells = [sorted_nodes[idx] for idx in best_overall_route]
    
    # Decision trace logging
    decision_trace = []
    for i in range(len(best_overall_target_seq) - 1):
        u_idx = best_overall_target_seq[i]
        v_idx = best_overall_target_seq[i+1]
        cache = cost_cache.get((u_idx, v_idx))
        decision_trace.append({
            "from_node": int(u_idx),
            "selected_target": int(v_idx),
            "path_found": cache is not None,
            "path_nodes": [int(n) for n in cache["path"]] if cache is not None else [],
            "action": "return_to_depot" if v_idx == depot_node_idx and i == len(best_overall_target_seq) - 2 else "visit_delivery"
        })
        
    unvisited_remaining = list(set(delivery_node_indices) - set(best_overall_target_seq))
    
    # Store final summary metrics
    results = {
        "route_nodes": best_overall_route,
        "route_cells": [[int(c[0]), int(c[1])] for c in route_cells],
        "decision_trace": decision_trace,
        "unvisited_remaining": [int(n) for n in unvisited_remaining],
        "total_duration_sec": float(best_overall_metrics["duration_sec"]),
        "total_distance_km": float(best_overall_metrics["distance_km"]),
        "avg_congestion": float(best_overall_metrics["avg_congestion"]),
        "fuel_l": float(best_overall_metrics["fuel_l"]),
        "co2_g": float(best_overall_metrics["co2_g"]),
        
        # Meta diagnostics
        "objective_value": float(best_overall_objective),
        "convergence_history": convergence_history,
        "pheromone_history": pheromone_matrix_history,
        "node_labels": [f"N_{n}" if n != depot_node_idx else "Depot" for n in aco_nodes]
    }
    
    return results
