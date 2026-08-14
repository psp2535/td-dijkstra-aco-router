import os
import json
import pandas as pd
import numpy as np
import torch
from src.utils.helpers import haversine_distance

def prepare_gnn_graph(processed_dir, config):
    """
    Orchestrate GNN graph preparation:
    1. Load processed clean trips, grid stats, nodes dictionary, and edges.
    2. Map grid cells to sequential node IDs.
    3. Construct node features (average speed, average congestion, points count, transition density).
    4. Construct edge features (centroid distance, speed, duration, congestion-aware weight).
    5. Construct PyTorch adjacency structures (edge_index list, dense adjacency matrix).
    6. Compute graph statistics.
    
    Returns:
        graph_data: dict of PyTorch tensors (x, edge_index, edge_attr, adj_matrix)
        stats: dict of graph statistics
    """
    # File paths
    nodes_json_path = os.path.join(processed_dir, "graph_nodes.json")
    edges_csv_path = os.path.join(processed_dir, "graph_edges.csv")
    congestion_csv_path = os.path.join(processed_dir, "grid_congestion_stats.csv")
    
    # Load data
    if not os.path.exists(nodes_json_path) or not os.path.exists(edges_csv_path):
        raise FileNotFoundError(
            f"Required graph files not found in {processed_dir}. Run baseline main.py first."
        )
        
    with open(nodes_json_path, 'r') as f:
        nodes_dict = json.load(f)
        
    edges_df = pd.read_csv(edges_csv_path)
    
    congestion_df = pd.read_csv(congestion_csv_path) if os.path.exists(congestion_csv_path) else pd.DataFrame()
    
    # 1. Map grid cell coordinates "row,col" to sequential node IDs
    # Parse keys to tuples and sort to ensure deterministic indexing
    sorted_nodes = sorted([tuple(map(int, k.split(','))) for k in nodes_dict.keys()])
    node_to_idx = {node: idx for idx, node in enumerate(sorted_nodes)}
    N = len(sorted_nodes)
    
    # 2. Prepare Node Features Matrix X (N x 4)
    # Features: [avg_speed, congestion_level, points_count_log, transition_density]
    # - avg_speed: mean speed inside cell
    # - congestion_level: average congestion level across hours
    # - points_count_log: log(1 + visits)
    # - transition_density: sum of incoming & outgoing transitions
    
    # Calculate transition density per cell from edges
    out_degrees = edges_df.groupby(["grid_row", "grid_col"])["transition_count"].sum().to_dict()
    in_degrees = edges_df.groupby(["next_row", "next_col"])["transition_count"].sum().to_dict()
    
    # Calculate average congestion level per cell
    avg_congestion = {}
    if not congestion_df.empty:
        # Group by grid row and col and compute mean congestion level
        cong_grp = congestion_df.groupby(["grid_row", "grid_col"])["congestion_level"].mean().to_dict()
        for k, v in cong_grp.items():
            avg_congestion[k] = v
            
    x_data = []
    bbox = config["preprocessing"]["bbox"]
    grid_dims = (config["spatial_grid"]["num_rows"], config["spatial_grid"]["num_cols"])
    
    for node in sorted_nodes:
        node_key = f"{node[0]},{node[1]}"
        node_meta = nodes_dict[node_key]
        
        # Feature 1: Average Speed (km/h)
        speed = float(node_meta.get("avg_speed_kmh", 0.0))
        
        # Feature 2: Congestion Level (0 = FreeFlow, 2 = Congested)
        cong = float(avg_congestion.get(node, 0.0))
        
        # Feature 3: Log-scaled Points Count (representing popularity/visits)
        pts = float(node_meta.get("points_count", 0.0))
        pts_log = float(np.log1p(pts))
        
        # Feature 4: Transition Density (Centrality in transition network)
        out_trans = out_degrees.get(node, 0)
        in_trans = in_degrees.get(node, 0)
        density = float(out_trans + in_trans)
        
        x_data.append([speed, cong, pts_log, density])
        
    x_tensor = torch.tensor(x_data, dtype=torch.float)
    
    # 3. Prepare Edge Index List (2 x E) & Adjacency Matrix (N x N)
    edge_sources = []
    edge_targets = []
    edge_feats = []
    
    # Get spatial step size to calculate cell centroids
    min_lat, min_lon, max_lat, max_lon = bbox
    num_rows, num_cols = grid_dims
    lat_step = (max_lat - min_lat) / num_rows
    lon_step = (max_lon - min_lon) / num_cols
    
    for _, edge in edges_df.iterrows():
        u = (int(edge["grid_row"]), int(edge["grid_col"]))
        v = (int(edge["next_row"]), int(edge["next_col"]))
        
        # Skip if nodes were filtered out and don't exist in node index map
        if u not in node_to_idx or v not in node_to_idx:
            continue
            
        u_idx = node_to_idx[u]
        v_idx = node_to_idx[v]
        
        edge_sources.append(u_idx)
        edge_targets.append(v_idx)
        
        # 4. Prepare Edge Features (E x 4)
        # Features: [distance, transition_speed, duration, congestion_aware_weight]
        
        # Feature 1: Transition Distance (Haversine distance between grid cell centroids)
        u_lat = min_lat + (u[0] + 0.5) * lat_step
        u_lon = min_lon + (u[1] + 0.5) * lon_step
        v_lat = min_lat + (v[0] + 0.5) * lat_step
        v_lon = min_lon + (v[1] + 0.5) * lon_step
        
        dist_m = haversine_distance(u_lat, u_lon, v_lat, v_lon)
        
        # Feature 2: Average Transition Speed (km/h)
        edge_speed = float(edge["avg_speed_kmh"])
        
        # Feature 3: Average Duration (seconds)
        edge_dur = float(edge["avg_duration_sec"])
        
        # Feature 4: Congestion-Aware Edge Weight (routing friction)
        # Fuel cost/time increases as surrounding nodes are congested. 
        # Weight = average_duration_sec * (1.0 + 0.5 * (source_congestion + target_congestion))
        u_cong = avg_congestion.get(u, 0.0)
        v_cong = avg_congestion.get(v, 0.0)
        cong_multiplier = 1.0 + 0.5 * (u_cong + v_cong)
        weight = edge_dur * cong_multiplier
        
        edge_feats.append([dist_m, edge_speed, edge_dur, weight])
        
    edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
    edge_attr = torch.tensor(edge_feats, dtype=torch.float)
    E = edge_attr.shape[0]
    
    # Construct dense Adjacency Matrix (N x N)
    adj_matrix = torch.zeros((N, N), dtype=torch.float)
    if E > 0:
        adj_matrix[edge_index[0], edge_index[1]] = 1.0
        
    # 5. Compute Graph Statistics
    # Average node degree
    avg_deg = E / N if N > 0 else 0.0
    
    # Connected Components (Weakly Connected Components ignoring edge directions)
    # Implement BFS components finder
    adj_undirected = {i: set() for i in range(N)}
    for src, dst in zip(edge_sources, edge_targets):
        adj_undirected[src].add(dst)
        adj_undirected[dst].add(src)
        
    visited = set()
    components = []
    for i in range(N):
        if i not in visited:
            comp = []
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for neighbor in adj_undirected[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(comp)
            
    num_components = len(components)
    comp_sizes = [len(c) for c in components]
    max_comp_size = max(comp_sizes) if comp_sizes else 0
    
    stats = {
        "num_nodes": N,
        "num_edges": E,
        "average_degree": avg_deg,
        "weakly_connected_components": num_components,
        "largest_component_size": max_comp_size,
        "isolated_nodes": comp_sizes.count(1)
    }
    
    graph_data = {
        "x": x_tensor,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "adj_matrix": adj_matrix
    }
    
    return graph_data, stats, sorted_nodes
