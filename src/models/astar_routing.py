import heapq
import numpy as np
from src.utils.helpers import haversine_distance

def get_cell_centroid(row, col, bbox, grid_dims):
    min_lat, min_lon, max_lat, max_lon = bbox
    num_rows, num_cols = grid_dims
    lat_step = (max_lat - min_lat) / num_rows
    lon_step = (max_lon - min_lon) / num_cols
    lat = min_lat + (row + 0.5) * lat_step
    lon = min_lon + (col + 0.5) * lon_step
    return lat, lon

def heuristic(u_idx, end_idx, sorted_nodes, bbox, grid_dims, w1, w2, w3, max_speed_mps=15.0):
    """
    Admissible A* heuristic combining distance, minimum duration, and free flow congestion cost.
    """
    if u_idx == end_idx:
        return 0.0
        
    u_cell = sorted_nodes[u_idx]
    v_cell = sorted_nodes[end_idx]
    
    lat_u, lon_u = get_cell_centroid(u_cell[0], u_cell[1], bbox, grid_dims)
    lat_v, lon_v = get_cell_centroid(v_cell[0], v_cell[1], bbox, grid_dims)
    
    dist = haversine_distance(lat_u, lon_u, lat_v, lon_v)
    min_dur = dist / max_speed_mps
    
    # Normalized heuristic values based on rough global bounds
    norm_dist = min(dist / 50000.0, 1.0)
    norm_dur = min(min_dur / 3600.0, 1.0)
    
    # Cost lower bound
    return w1 * norm_dur + w2 * norm_dist

def astar_weighted_pathfinder(start_idx, end_idx, weighted_adj_dict, N, sorted_nodes, bbox, grid_dims, w1, w2, w3):
    """
    A* pathfinder to minimize cumulative Cost(e) while tracking duration and distance.
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
    
    # Priority Queue stores tuples: (f_score, curr_cost, node_idx, duration, distance)
    # f_score = g_score (curr_cost) + h_score (heuristic)
    pq = [(heuristic(start_idx, end_idx, sorted_nodes, bbox, grid_dims, w1, w2, w3), 0.0, start_idx, 0.0, 0.0)]
    
    while pq:
        f_score, curr_cost, curr_node, curr_dur, curr_dist = heapq.heappop(pq)
        
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
                
                h_val = heuristic(neighbor, end_idx, sorted_nodes, bbox, grid_dims, w1, w2, w3)
                heapq.heappush(pq, (new_cost + h_val, new_cost, neighbor, new_dur, new_dist))
                
    if costs[end_idx] == float('inf'):
        return None, float('inf'), float('inf'), float('inf')
        
    path = []
    curr = end_idx
    while curr is not None:
        path.append(curr)
        curr = parents[curr]
    path.reverse()
    
    return path, costs[end_idx], durations[end_idx], distances[end_idx]
