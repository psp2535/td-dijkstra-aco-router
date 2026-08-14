import networkx as nx
import osmnx as ox
import time
import math
from src.utils.helpers import haversine_distance

def get_predicted_speed_at_time(cell, t, predictions_map, global_ratios, cell_ratios, free_flow_speed_mps, default_speed_mps):
    """
    Compute continuous, FIFO-compliant speed (in m/s) at fractional departure time t on a road segment,
    based on the grid cell's traffic predictions.
    """
    ff_speed = free_flow_speed_mps if free_flow_speed_mps > 0 else default_speed_mps

    fractional_hour = (t / 3600.0) % 24
    h1 = int(fractional_hour)
    h2 = (h1 + 1) % 24
    lambda_weight = fractional_hour - h1

    # Look up predicted levels for the grid cell mapped to this edge
    c1 = predictions_map.get((cell[0], cell[1], h1), 0)
    c2 = predictions_map.get((cell[0], cell[1], h2), 0)

    ratio1 = cell_ratios.get((cell[0], cell[1], c1), global_ratios.get(c1, 0.85))
    ratio2 = cell_ratios.get((cell[0], cell[1], c2), global_ratios.get(c2, 0.85))

    v1 = ff_speed * ratio1
    v2 = ff_speed * ratio2

    v_interp = (1.0 - lambda_weight) * v1 + lambda_weight * v2
    return max(v_interp, 0.556) # minimum ~2 km/h


def osmnx_dijkstra_static(G, start_node, end_node):
    """
    Standard Static Dijkstra over OSMnx graph. Weight is 'static_time'.
    Returns path (list of node IDs), total duration, total distance.
    """
    try:
        path = nx.shortest_path(G, source=start_node, target=end_node, weight='static_time')
        duration = 0.0
        distance = 0.0
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i+1]
            edge_data = G.get_edge_data(u, v)
            if edge_data:
                # OSMnx graphs can be MultiDiGraphs (multiple edges between same nodes). Take the first (key=0).
                data = edge_data[min(edge_data.keys())]
                duration += data.get('static_time', 0.0)
                distance += data.get('length', 0.0)
        return path, duration, distance
    except nx.NetworkXNoPath:
        return None, float('inf'), float('inf')


def osmnx_dijkstra_td(G, start_node, end_node, departure_time_sec, predictions_map, global_ratios, cell_ratios, cell_ff_speeds):
    """
    Time-Dependent Dijkstra over OSMnx graph.
    Modifies edge traversal times dynamically based on traffic mapped to the edge's grid_cell.
    Returns path (list of node IDs), total duration, total distance.
    """
    if start_node == end_node:
        return [start_node], 0.0, 0.0
        
    import heapq
    
    # We will use a dict for visited nodes to track the earliest arrival time
    arrival_times = {start_node: float(departure_time_sec)}
    distances = {start_node: 0.0}
    parents = {start_node: None}
    
    # Priority Queue stores tuples: (arrival_time, current_node, cumulative_distance)
    pq = [(float(departure_time_sec), start_node, 0.0)]
    
    while pq:
        curr_time, curr_node, curr_dist = heapq.heappop(pq)
        
        if curr_node == end_node:
            break
            
        if curr_time > arrival_times.get(curr_node, float('inf')):
            continue
            
        # Iterate over out-edges
        if curr_node not in G:
            continue
            
        for neighbor in G.successors(curr_node):
            edge_data = G.get_edge_data(curr_node, neighbor)
            if not edge_data: continue
            data = edge_data[min(edge_data.keys())]
            
            length = data.get('length', 10.0)
            cell = data.get('grid_cell', None)
            
            # Default speed limit derived from static_time
            static_t = data.get('static_time', length / 8.33)
            default_speed = length / static_t if static_t > 0 else 8.33
            
            if cell is not None:
                ff_speed_kmh = cell_ff_speeds.get(cell, 0.0)
                ff_speed_mps = ff_speed_kmh / 3.6 if ff_speed_kmh > 0 else 0.0
                speed_mps = get_predicted_speed_at_time(cell, curr_time, predictions_map, global_ratios, cell_ratios, ff_speed_mps, default_speed)
            else:
                speed_mps = default_speed
                
            travel_time = length / speed_mps if speed_mps > 0 else float('inf')
            
            new_time = curr_time + travel_time
            new_dist = curr_dist + length
            
            if new_time < arrival_times.get(neighbor, float('inf')):
                arrival_times[neighbor] = new_time
                distances[neighbor] = new_dist
                parents[neighbor] = curr_node
                heapq.heappush(pq, (new_time, neighbor, new_dist))
                
    if end_node not in arrival_times:
        return None, float('inf'), float('inf')
        
    path = []
    curr = end_node
    while curr is not None:
        path.append(curr)
        curr = parents[curr]
    path.reverse()
    
    return path, arrival_times[end_node] - departure_time_sec, distances[end_node]

def extract_path_segments(G, path, is_td=False, start_time_sec=0.0, predictions_map=None, global_ratios=None, cell_ratios=None, free_flow_speeds=None):
    """
    Given a list of OSMnx node IDs, return the exact sequence of segments with their coordinates, length, duration, and road name.
    """
    segments = []
    if not path or len(path) < 2: return segments
    
    curr_time = start_time_sec
    
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i+1]
        edge_data = G.get_edge_data(u, v)
        if not edge_data: continue
        data = edge_data[min(edge_data.keys())]
        
        name = data.get('name', 'Unnamed Road')
        if isinstance(name, list): name = name[0]
        
        length = data.get('length', 10.0)
        
        if is_td and predictions_map is not None:
            cell = data.get('grid_cell')
            static_t = data.get('static_time', length/8.33)
            default_speed = length / static_t if static_t > 0 else 8.33
            ff_speed_kmh = free_flow_speeds.get(cell, 0.0) if cell else 0.0
            ff_speed_mps = ff_speed_kmh / 3.6 if ff_speed_kmh > 0 else 0.0
            
            speed_mps = get_predicted_speed_at_time(
                cell, curr_time, predictions_map, 
                global_ratios, cell_ratios, 
                ff_speed_mps, default_speed
            )
            dur = length / speed_mps if speed_mps > 0 else length / 8.33
        else:
            dur = data.get('static_time', length/8.33)
            speed_mps = length / dur if dur > 0 else 8.33
            
        curr_time += dur
        
        coords = []
        if 'geometry' in data:
            geom_coords = list(data['geometry'].coords)
            for lon, lat in geom_coords:
                coords.append([lat, lon])
        else:
            coords.append([G.nodes[u]['y'], G.nodes[u]['x']])
            coords.append([G.nodes[v]['y'], G.nodes[v]['x']])
            
        segments.append({
            "name": name,
            "length_m": float(length),
            "duration_sec": float(dur),
            "speed_kmh": float(speed_mps * 3.6),
            "coords": coords
        })
        
    return segments

