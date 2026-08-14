import heapq
import numpy as np
import pandas as pd
import torch
from src.utils.helpers import haversine_distance

def dijkstra_pathfinder(start_idx, end_idx, adj_dict, N):
    """
    Dijkstra's shortest path algorithm implemented in pure Python.
    Minimizes edge travel time (avg_duration_sec).
    
    adj_dict: dict mapping node_idx -> list of tuples (neighbor_idx, duration, distance)
    N: total number of nodes
    
    Returns:
        path: list of node indices from start_idx to end_idx (inclusive)
        duration: total travel duration in seconds
        distance: total travel distance in meters
    """
    if start_idx == end_idx:
        return [start_idx], 0.0, 0.0
        
    distances = {i: float('inf') for i in range(N)}
    durations = {i: float('inf') for i in range(N)}
    parents = {i: None for i in range(N)}
    
    distances[start_idx] = 0.0
    durations[start_idx] = 0.0
    
    # Priority queue stores tuples: (duration, node_idx, cumulative_distance)
    pq = [(0.0, start_idx, 0.0)]
    
    while pq:
        curr_dur, curr_node, curr_dist = heapq.heappop(pq)
        
        if curr_node == end_idx:
            break
            
        if curr_dur > durations[curr_node]:
            continue
            
        for neighbor, duration, distance in adj_dict.get(curr_node, []):
            new_dur = curr_dur + duration
            new_dist = curr_dist + distance
            
            if new_dur < durations[neighbor]:
                durations[neighbor] = new_dur
                distances[neighbor] = new_dist
                parents[neighbor] = curr_node
                heapq.heappush(pq, (new_dur, neighbor, new_dist))
                
    if durations[end_idx] == float('inf'):
        return None, float('inf'), float('inf')
        
    # Reconstruct path
    path = []
    curr = end_idx
    while curr is not None:
        path.append(curr)
        curr = parents[curr]
    path.reverse()
    
    return path, durations[end_idx], distances[end_idx]

def greedy_route_planner(depot_node_idx, delivery_node_indices, embeddings, adj_dict, N, return_to_depot=True):
    """
    Plan a GNN-guided Greedy routing sequence across multiple delivery nodes.
    
    For a current node, select the next target u* from unvisited deliveries U:
    u* = argmin_{u in U} || Embedding(v) - Embedding(u) ||_2 (reproducing Eq. 30)
    
    Connect consecutive targets using Dijkstra-lite pathfinder.
    """
    embeddings_np = embeddings.numpy()
    current_node = depot_node_idx
    unvisited = set(delivery_node_indices)
    
    full_route = [current_node]
    decision_trace = []
    
    total_duration = 0.0
    total_distance_m = 0.0
    
    while unvisited:
        # 1. Compute embedding distance to all remaining unvisited targets
        candidates = list(unvisited)
        h_current = embeddings_np[current_node]
        h_candidates = embeddings_np[candidates]
        
        # L2 distances in GNN embedding space
        dists = np.linalg.norm(h_candidates - h_current, ord=2, axis=1)
        
        # Sort candidates by GNN embedding proximity
        sorted_indices = np.argsort(dists)
        sorted_candidates = [candidates[i] for i in sorted_indices]
        sorted_dists = [float(dists[i]) for i in sorted_indices]
        
        path_found = False
        target_node = None
        chosen_path = None
        chosen_dur = 0.0
        chosen_dist = 0.0
        
        # 2. Try to route to the closest candidates in order of GNN proximity
        for cand, dist in zip(sorted_candidates, sorted_dists):
            # Check graph path using Dijkstra
            path, dur, dist_m = dijkstra_pathfinder(current_node, cand, adj_dict, N)
            if path is not None:
                path_found = True
                target_node = cand
                chosen_path = path
                chosen_dur = dur
                chosen_dist = dist_m
                break
                
        # Record trace of the decision process for diagnostics
        decision_trace.append({
            "from_node": int(current_node),
            "candidates_evaluated": [int(c) for c in sorted_candidates],
            "embedding_distances": sorted_dists,
            "path_found": path_found,
            "selected_target": int(target_node) if path_found else None,
            "path_nodes": [int(n) for n in chosen_path] if path_found else []
        })
        
        if not path_found:
            # Cannot reach any of the remaining delivery locations; terminate greedy search
            break
            
        # Append path to the route (exclude first node to avoid duplicate step-over nodes)
        full_route.extend(chosen_path[1:])
        unvisited.remove(target_node)
        current_node = target_node
        
        total_duration += chosen_dur
        total_distance_m += chosen_dist
        
    # 3. Return to depot option
    depot_return_path = []
    if return_to_depot and current_node != depot_node_idx:
        path, dur, dist_m = dijkstra_pathfinder(current_node, depot_node_idx, adj_dict, N)
        if path is not None:
            full_route.extend(path[1:])
            total_duration += dur
            total_distance_m += dist_m
            depot_return_path = path
            
        decision_trace.append({
            "from_node": int(current_node),
            "action": "return_to_depot",
            "path_found": path is not None,
            "path_nodes": [int(n) for n in path] if path is not None else []
        })
        
    return {
        "route_nodes": full_route,
        "decision_trace": decision_trace,
        "unvisited_remaining": [int(u) for u in unvisited],
        "total_duration_sec": total_duration,
        "total_distance_km": total_distance_m / 1000.0
    }

def evaluate_route(route_nodes, congestion_df, sorted_nodes):
    """
    Calculate average congestion scores of cells visited along the route.
    """
    if not route_nodes:
        return 0.0
        
    # Map node indices back to grid cell coordinates (row, col)
    route_cells = [sorted_nodes[idx] for idx in route_nodes]
    
    # Calculate average congestion level of visited cells
    avg_congestion = {}
    if not congestion_df.empty:
        cong_grp = congestion_df.groupby(["grid_row", "grid_col"])["congestion_level"].mean().to_dict()
        for k, v in cong_grp.items():
            avg_congestion[k] = v
            
    cong_levels = [avg_congestion.get(cell, 0.0) for cell in route_cells]
    mean_cong = float(np.mean(cong_levels)) if cong_levels else 0.0
    
    return mean_cong
