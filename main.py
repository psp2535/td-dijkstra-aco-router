import os
import argparse
import glob
import pandas as pd
import numpy as np
import json
from src.utils.helpers import load_config, setup_logging
from src.preprocessing.cleaning import load_t_drive_file, clean_trajectory
from src.preprocessing.segmentation import segment_trips
from src.features.metrics import calculate_kinematics, compute_fuel_and_emissions
from src.features.spatial_grid import compute_grid_congestion
from src.features.vehicle import generate_synthetic_fleet
from src.graph.grid_graph import build_grid_graph


def parse_args():
    parser = argparse.ArgumentParser(
        description="IM-VRM Baseline Preprocessing Pipeline for T-Drive Dataset"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default=None, 
        help="Path to YAML configuration file."
    )
    parser.add_argument(
        "--num-taxis", 
        type=int, 
        default=50, 
        help="Number of taxi trajectory files to process (default: 50, use -1 for all)."
    )
    parser.add_argument(
        "--step", 
        type=str, 
        choices=["all", "preprocess", "congestion", "graph", "fleet", "gnn_prep", "gnn_embed", "greedy_route", "tcfmu", "route_opt", "aco_route"], 
        default="all", 
        help="Run a specific pipeline step (default: all)."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load configuration and setup logging
    config = load_config(args.config)
    
    raw_dir = config["data"]["raw_dir"]
    processed_dir = config["data"]["processed_dir"]
    outputs_dir = config["data"]["outputs_dir"]
    
    # Ensure directories exist
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    
    logger = setup_logging(log_dir=outputs_dir)
    logger.info("Starting IM-VRM baseline pipeline execution.")
    logger.info(f"Configuration loaded. Raw dir: {raw_dir}, Processed dir: {processed_dir}")
    
    # Locate raw T-Drive files
    search_path = os.path.join(raw_dir, "*.txt")
    raw_files = sorted(glob.glob(search_path))
    
    if not raw_files:
        logger.error(f"No trajectory files (.txt) found in {raw_dir}. Please verify the path.")
        return
        
    num_to_process = len(raw_files) if args.num_taxis == -1 else min(args.num_taxis, len(raw_files))
    logger.info(f"Found {len(raw_files)} total taxi files. Processing target: {num_to_process} files.")
    
    files_to_process = raw_files[:num_to_process]
    
    # DataFrame to accumulate processed trajectories
    all_clean_trips = []
    
    # Phase 1: Preprocess and Feature Engineering
    if args.step in ["all", "preprocess"]:
        logger.info("=== STEP 1: Preprocessing & Feature Engineering ===")
        bbox = config["preprocessing"]["bbox"]
        speed_limit = config["preprocessing"]["speed_limit_kmh"]
        gap_threshold = config["preprocessing"]["inactivity_threshold_sec"]
        
        logger.info(f"Spatial Box filter: {bbox}")
        logger.info(f"Speed limit threshold: {speed_limit} km/h")
        logger.info(f"Trip gap threshold: {gap_threshold} seconds")
        
        for idx, filepath in enumerate(files_to_process):
            file_basename = os.path.basename(filepath)
            
            try:
                # Load
                df = load_t_drive_file(filepath)
                if df.empty:
                    continue
                
                # Clean
                df_clean = clean_trajectory(df, bbox, speed_limit)
                if len(df_clean) < config["preprocessing"]["min_point_count"]:
                    continue
                
                # Trip Segment
                df_trips = segment_trips(df_clean, gap_threshold)
                
                # Kinematics (distance, time, velocity, acceleration)
                df_features = calculate_kinematics(df_trips)
                
                # Green metrics (fuel and CO2 for light vs. heavy configurations)
                light_config = config["vehicle_params"]["light_duty"]
                heavy_config = config["vehicle_params"]["heavy_duty"]
                
                # Compute emissions for light vehicle representation
                fuel_light, co2_light = compute_fuel_and_emissions(
                    df_features["speed_kmh"].values,
                    df_features["delta_dist_m"].values,
                    df_features["delta_time_sec"].values,
                    light_config
                )
                df_features["fuel_light_l"] = fuel_light
                df_features["co2_light_g"] = co2_light
                
                # Compute emissions for heavy vehicle representation
                fuel_heavy, co2_heavy = compute_fuel_and_emissions(
                    df_features["speed_kmh"].values,
                    df_features["delta_dist_m"].values,
                    df_features["delta_time_sec"].values,
                    heavy_config
                )
                df_features["fuel_heavy_l"] = fuel_heavy
                df_features["co2_heavy_g"] = co2_heavy
                
                all_clean_trips.append(df_features)
                
                if (idx + 1) % 10 == 0 or (idx + 1) == num_to_process:
                    logger.info(f"Processed {idx + 1}/{num_to_process} files...")
                    
            except Exception as e:
                logger.warning(f"Failed to process file {file_basename}: {str(e)}")
                
        if all_clean_trips:
            merged_trips_df = pd.concat(all_clean_trips, ignore_index=True)
            output_csv = os.path.join(processed_dir, "clean_trips.csv")
            merged_trips_df.to_csv(output_csv, index=False)
            logger.info(f"Saved merged trajectories: {len(merged_trips_df)} points, {merged_trips_df['trip_id'].nunique()} trips.")
            logger.info(f"File saved to: {output_csv}")
        else:
            logger.error("No valid trajectory points remained after filtering.")
            return
    else:
        # Load preprocessed data from file for subsequent steps
        output_csv = os.path.join(processed_dir, "clean_trips.csv")
        if os.path.exists(output_csv):
            logger.info(f"Loading preprocessed trajectory data from {output_csv}")
            merged_trips_df = pd.read_csv(output_csv, parse_dates=["timestamp"])
        else:
            logger.error(f"Preprocessed data not found at {output_csv}. Please run --step preprocess first.")
            return

    # Phase 2: Congestion Analysis
    if args.step in ["all", "congestion"]:
        logger.info("=== STEP 2: Spatial Grid Congestion Labeling ===")
        bbox = config["preprocessing"]["bbox"]
        grid_dims = (config["spatial_grid"]["num_rows"], config["spatial_grid"]["num_cols"])
        
        logger.info(f"Constructing grid of dimensions: {grid_dims}")
        grid_stats, cell_free_flow = compute_grid_congestion(
            merged_trips_df, 
            bbox, 
            grid_dims, 
            config=config
        )
        
        if not grid_stats.empty:
            stats_csv = os.path.join(processed_dir, "grid_congestion_stats.csv")
            ff_csv = os.path.join(processed_dir, "cell_free_flow_speeds.csv")
            
            grid_stats.to_csv(stats_csv, index=False)
            cell_free_flow.to_csv(ff_csv, index=False)
            
            logger.info(f"Congestion profiling complete. Unique cells measured: {len(cell_free_flow)}")
            logger.info(f"Congestion records saved to: {stats_csv}")
            
            # Print congestion summary
            level_counts = grid_stats["congestion_level"].value_counts()
            logger.info("Congestion level breakdown across binned hour intervals:")
            for lvl, cnt in level_counts.items():
                lbl = ["FreeFlow", "Moderate", "Congested"][lvl]
                logger.info(f"  Level {lvl} ({lbl}): {cnt} entries ({cnt/len(grid_stats)*100:.1f}%)")
        else:
            logger.warning("Could not calculate congestion labels due to insufficient speed records.")

    # Phase 3: Transition Graph Construction
    if args.step in ["all", "graph"]:
        logger.info("=== STEP 3: Grid Transition Graph Construction ===")
        grid_dims = (config["spatial_grid"]["num_rows"], config["spatial_grid"]["num_cols"])
        bbox = config["preprocessing"]["bbox"]
        
        nodes, edges = build_grid_graph(
            merged_trips_df, 
            grid_dims, 
            bbox, 
            config=config
        )
        
        if nodes:
            # Save nodes as json (keys are tuples, convert to string representation for json compatibility)
            nodes_json_path = os.path.join(processed_dir, "graph_nodes.json")
            nodes_str_keys = {f"{k[0]},{k[1]}": v for k, v in nodes.items()}
            with open(nodes_json_path, 'w') as f:
                json.dump(nodes_str_keys, f, indent=2)
                
            # Save edges as csv
            edges_csv_path = os.path.join(processed_dir, "graph_edges.csv")
            edges.to_csv(edges_csv_path, index=False)
            
            logger.info(f"Graph construction completed. Nodes: {len(nodes)}, Edges: {len(edges)}")
            logger.info(f"Nodes saved to: {nodes_json_path}")
            logger.info(f"Edges saved to: {edges_csv_path}")
        else:
            logger.warning("Could not construct grid transition graph due to empty trajectory data.")

    # Phase 4: Fleet Metadata Simulation parameters
    if args.step in ["all", "fleet"]:
        logger.info("=== STEP 4: Synthetic Vehicle Fleet Generation ===")
        # Generate fleet
        fleet, depots = generate_synthetic_fleet(num_vehicles=args.num_taxis, config=config)
        
        fleet_json_path = os.path.join(processed_dir, "synthetic_fleet.json")
        with open(fleet_json_path, 'w') as f:
            json.dump({
                "depots": depots,
                "fleet": fleet
            }, f, indent=2)
            
        logger.info(f"Generated {len(fleet)} synthetic vehicles across {len(depots)} depots.")
        logger.info(f"Fleet metadata saved to: {fleet_json_path}")

    # Phase 5: GNN Graph Data Preparation
    if args.step in ["all", "gnn_prep"]:
        logger.info("=== STEP 5: GNN Graph Data & Node/Edge Feature Preparation ===")
        try:
            import torch
            from src.graph.gnn_features import prepare_gnn_graph
            
            graph_data, stats, sorted_nodes = prepare_gnn_graph(processed_dir, config)
            
            # Save graph data as PyTorch .pt file
            gnn_pt_path = os.path.join(processed_dir, "gnn_graph_data.pt")
            torch.save(graph_data, gnn_pt_path)
            
            logger.info("GNN Graph preparation complete. Graph stats:")
            logger.info(f"  Nodes: {stats['num_nodes']}")
            logger.info(f"  Edges: {stats['num_edges']}")
            logger.info(f"  Average Degree: {stats['average_degree']:.3f}")
            logger.info(f"  Weakly Connected Components: {stats['weakly_connected_components']}")
            logger.info(f"  Largest Component Size: {stats['largest_component_size']}")
            logger.info(f"  Isolated Nodes: {stats['isolated_nodes']}")
            logger.info(f"GNN graph tensors saved to: {gnn_pt_path}")
            
            # Save stats to a JSON file for inspection
            stats_json_path = os.path.join(processed_dir, "gnn_graph_stats.json")
            with open(stats_json_path, 'w') as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Graph statistics report saved to: {stats_json_path}")
        except Exception as e:
            logger.error(f"Failed to prepare GNN graph features: {str(e)}")

    # Phase 6: GNN Node Embedding Generation
    if args.step in ["all", "gnn_embed"]:
        logger.info("=== STEP 6: GNN Node Embedding Generation ===")
        try:
            import torch
            from src.models.gnn_embedding import train_gnn_embeddings
            from src.utils.gnn_diagnostics import analyze_and_visualize_embeddings
            
            # Load GNN graph data
            gnn_pt_path = os.path.join(processed_dir, "gnn_graph_data.pt")
            if not os.path.exists(gnn_pt_path):
                logger.error(f"GNN Graph data not found at {gnn_pt_path}. Please run --step gnn_prep first.")
                return
                
            graph_data = torch.load(gnn_pt_path)
            logger.info(f"Loaded graph tensors. Running GNN training for 100 epochs (Input: 4, Hidden: 32, Output: 64)...")
            
            # Run training to extract embeddings
            embeddings, loss_history = train_gnn_embeddings(graph_data, epochs=100, lr=0.01, seed=config.get("seed", 42))
            
            # Save node embeddings matrix
            embed_pt_path = os.path.join(processed_dir, "node_embeddings.pt")
            torch.save(embeddings, embed_pt_path)
            logger.info(f"Final node embeddings saved to: {embed_pt_path}")
            
            # Load sorted node keys to map coordinates correctly
            nodes_json_path = os.path.join(processed_dir, "graph_nodes.json")
            with open(nodes_json_path, 'r') as f:
                nodes_dict = json.load(f)
            sorted_nodes = sorted([tuple(map(int, k.split(','))) for k in nodes_dict.keys()])
            
            # Run diagnostics and visualization
            congestion_csv_path = os.path.join(processed_dir, "grid_congestion_stats.csv")
            stats, plot_path = analyze_and_visualize_embeddings(
                embeddings, 
                sorted_nodes, 
                congestion_csv_path, 
                outputs_dir
            )
            
            logger.info("GNN Embedding Generation Diagnostics:")
            logger.info(f"  Shape: {stats['embedding_shape']}")
            logger.info(f"  Contains NaN: {stats['has_nans']}")
            logger.info(f"  L2 Norm mean: {stats['l2_norm_stats']['mean']:.4f} (std: {stats['l2_norm_stats']['std']:.4f})")
            logger.info(f"  PCA explained variance: {stats['pca_explained_variance_ratio']}")
            logger.info(f"Visualization plot saved to: {plot_path}")
            logger.info(f"Embedding statistics saved to: {os.path.join(outputs_dir, 'embedding_statistics.json')}")
            
        except Exception as e:
            logger.error(f"Failed to generate GNN embeddings: {str(e)}")

    # Phase 7: GNN-Guided Greedy Route Selection
    if args.step in ["all", "greedy_route"]:
        logger.info("=== STEP 7: GNN-Guided Greedy Route Selection (Dijkstra-Lite) ===")
        try:
            import torch
            from src.models.greedy_routing import greedy_route_planner, evaluate_route
            from src.utils.routing_viz import plot_greedy_route
            from src.utils.helpers import haversine_distance
            
            # Load embeddings
            embed_pt_path = os.path.join(processed_dir, "node_embeddings.pt")
            if not os.path.exists(embed_pt_path):
                logger.error(f"Node embeddings not found at {embed_pt_path}. Please run --step gnn_embed first.")
                return
            embeddings = torch.load(embed_pt_path)
            N = embeddings.shape[0]
            
            # Load nodes json and parse sorted nodes
            nodes_json_path = os.path.join(processed_dir, "graph_nodes.json")
            with open(nodes_json_path, 'r') as f:
                nodes_dict = json.load(f)
            sorted_nodes = sorted([tuple(map(int, k.split(','))) for k in nodes_dict.keys()])
            node_to_idx = {node: idx for idx, node in enumerate(sorted_nodes)}
            
            # Load edges csv
            edges_csv_path = os.path.join(processed_dir, "graph_edges.csv")
            edges_df = pd.read_csv(edges_csv_path)
            
            # Load congestion stats
            congestion_csv_path = os.path.join(processed_dir, "grid_congestion_stats.csv")
            congestion_df = pd.read_csv(congestion_csv_path) if os.path.exists(congestion_csv_path) else pd.DataFrame()
            
            # Load fleet config
            fleet_json_path = os.path.join(processed_dir, "synthetic_fleet.json")
            if not os.path.exists(fleet_json_path):
                logger.error(f"Synthetic fleet metadata not found at {fleet_json_path}. Please run --step fleet first.")
                return
            with open(fleet_json_path, 'r') as f:
                fleet_data = json.load(f)
                
            fleet = fleet_data["fleet"]
            depots = fleet_data["depots"]
            
            # Build transition adjacency dictionary with (duration, centroid_distance)
            # Edge weights = avg_duration_sec
            bbox = config["preprocessing"]["bbox"]
            grid_dims = (config["spatial_grid"]["num_rows"], config["spatial_grid"]["num_cols"])
            min_lat, min_lon, max_lat, max_lon = bbox
            num_rows, num_cols = grid_dims
            lat_step = (max_lat - min_lat) / num_rows
            lon_step = (max_lon - min_lon) / num_cols
            
            adj_dict = {}
            for _, edge in edges_df.iterrows():
                u = (int(edge["grid_row"]), int(edge["grid_col"]))
                v = (int(edge["next_row"]), int(edge["next_col"]))
                
                if u in node_to_idx and v in node_to_idx:
                    u_idx = node_to_idx[u]
                    v_idx = node_to_idx[v]
                    
                    # Compute geodetic distance between centroids
                    u_lat = min_lat + (u[0] + 0.5) * lat_step
                    u_lon = min_lon + (u[1] + 0.5) * lon_step
                    v_lat = min_lat + (v[0] + 0.5) * lat_step
                    v_lon = min_lon + (v[1] + 0.5) * lon_step
                    dist_m = haversine_distance(u_lat, u_lon, v_lat, v_lon)
                    
                    duration = float(edge["avg_duration_sec"])
                    
                    if u_idx not in adj_dict:
                        adj_dict[u_idx] = []
                    adj_dict[u_idx].append((v_idx, duration, dist_m))
                    
            logger.info(f"Constructed routing network: {N} active cell nodes, {len(edges_df)} transition edges.")
            
            # Build set of grid cells that have at least one outgoing edge
            nodes_with_out_edges = set(
                edges_df["grid_row"].astype(str) + "," + edges_df["grid_col"].astype(str)
            )
            
            # Helper to map any cell coordinate to the closest visited graph node with out-degree > 0
            def find_closest_active_node(row, col):
                min_d = float('inf')
                best_idx = 0
                for idx, node in enumerate(sorted_nodes):
                    node_key = f"{node[0]},{node[1]}"
                    if node_key not in nodes_with_out_edges:
                        continue
                    d = abs(node[0] - row) + abs(node[1] - col)
                    if d < min_d:
                        min_d = d
                        best_idx = idx
                
                # Fallback to any node if none has outgoing edges (e.g. extremely small graph)
                if min_d == float('inf'):
                    for idx, node in enumerate(sorted_nodes):
                        d = abs(node[0] - row) + abs(node[1] - col)
                        if d < min_d:
                            min_d = d
                            best_idx = idx
                return best_idx
                
            routes_report = {}
            metrics_list = []
            
            # Seed generator for deterministic delivery cell requests selection
            np.random.seed(42)
            
            logger.info("Executing Greedy Search routing on synthetic vehicle fleet...")
            for idx, vehicle in enumerate(fleet):
                v_id = vehicle["vehicle_id"]
                depot_row = vehicle["depot_grid_row"]
                depot_col = vehicle["depot_grid_col"]
                
                # Map depot cell to closest visited grid node index
                depot_idx = find_closest_active_node(depot_row, depot_col)
                
                # Generate 5 random distinct delivery node indices from active cells (different from depot)
                available_nodes = [i for i in range(N) if i != depot_idx]
                if len(available_nodes) >= 5:
                    delivery_indices = list(np.random.choice(available_nodes, size=5, replace=False))
                else:
                    delivery_indices = available_nodes
                    
                # Plan route
                plan = greedy_route_planner(
                    depot_idx, 
                    delivery_indices, 
                    embeddings, 
                    adj_dict, 
                    N, 
                    return_to_depot=True
                )
                
                # Calculate congestion metrics
                route_nodes = plan["route_nodes"]
                avg_congestion = evaluate_route(route_nodes, congestion_df, sorted_nodes)
                
                # Save route details
                route_cells = [sorted_nodes[node_idx] for node_idx in route_nodes]
                routes_report[v_id] = {
                    "vehicle_id": v_id,
                    "type": vehicle["type"],
                    "depot_id": vehicle["depot_id"],
                    "depot_cell": [int(depot_row), int(depot_col)],
                    "mapped_depot_cell": list(sorted_nodes[depot_idx]),
                    "delivery_cells": [list(sorted_nodes[i]) for i in delivery_indices],
                    "route_cells": [[int(c[0]), int(c[1])] for c in route_cells],
                    "decision_trace": plan["decision_trace"],
                    "unvisited_remaining": plan["unvisited_remaining"],
                    "total_distance_km": float(plan["total_distance_km"]),
                    "total_duration_sec": float(plan["total_duration_sec"]),
                    "avg_congestion": avg_congestion
                }
                
                metrics_list.append({
                    "vehicle_id": v_id,
                    "vehicle_type": vehicle["type"],
                    "depot_id": vehicle["depot_id"],
                    "distance_km": float(plan["total_distance_km"]),
                    "duration_sec": float(plan["total_duration_sec"]),
                    "route_length_nodes": len(route_nodes),
                    "avg_congestion": avg_congestion,
                    "unvisited_count": len(plan["unvisited_remaining"])
                })
                
            # Save routes report as json
            routes_json_path = os.path.join(processed_dir, "baseline_routes.json")
            with open(routes_json_path, 'w') as f:
                json.dump(routes_report, f, indent=2)
                
            # Save metrics report as csv
            metrics_df = pd.DataFrame(metrics_list)
            metrics_csv_path = os.path.join(outputs_dir, "greedy_metrics.csv")
            metrics_df.to_csv(metrics_csv_path, index=False)
            
            logger.info(f"Greedy routing completed for {len(fleet)} vehicles.")
            logger.info(f"Routes saved to: {routes_json_path}")
            logger.info(f"Performance metrics saved to: {metrics_csv_path}")
            
            # Print average performance metrics
            avg_dist = metrics_df["distance_km"].mean()
            avg_dur = metrics_df["duration_sec"].mean()
            avg_unvisited = metrics_df["unvisited_count"].mean()
            logger.info(f"Average Route Distance: {avg_dist:.3f} km")
            logger.info(f"Average Route Duration: {avg_dur:.1f} seconds ({avg_dur/60:.1f} mins)")
            logger.info(f"Average Unvisited Node Gaps: {avg_unvisited:.1f} nodes")
            
            # 5. Visualize a demonstration route (e.g. for vehicle_001)
            demo_veh = "vehicle_001"
            if demo_veh in routes_report:
                demo_data = routes_report[demo_veh]
                demo_route = [tuple(c) for c in demo_data["route_cells"]]
                demo_depot = tuple(demo_data["mapped_depot_cell"])
                demo_deliveries = [tuple(c) for c in demo_data["delivery_cells"]]
                plot_path = os.path.join(outputs_dir, "route_demo.png")
                
                plot_greedy_route(grid_dims, demo_route, demo_depot, demo_deliveries, plot_path)
                logger.info(f"Saved route demonstration visualization for {demo_veh} to: {plot_path}")
                
        except Exception as e:
            logger.error(f"Failed to execute GNN-guided Greedy routing: {str(e)}")

    # Phase 8: XGBoost Traffic Congestion Prediction (TCFMu)
    if args.step in ["all", "tcfmu"]:
        logger.info("=== STEP 8: XGBoost Traffic Congestion Prediction (TCFMu) ===")
        try:
            from src.models.tcfmu import prepare_congestion_dataset, train_tcfmu, evaluate_and_plot_tcfmu
            
            # Prepare dataset
            X, y = prepare_congestion_dataset(processed_dir)
            logger.info(f"Prepared congestion prediction dataset. Features shape: {X.shape}, Target shape: {y.shape}")
            
            # Train model
            model, X_train, y_train, X_test, y_test = train_tcfmu(X, y, config, outputs_dir)
            
            # Evaluate model
            stats = evaluate_and_plot_tcfmu(model, X_train, X_test, y_test, outputs_dir)
            
            # Save test predictions to CSV
            y_pred = model.predict(X_test)
            predictions_df = X_test.copy()
            predictions_df["true_congestion_level"] = y_test
            predictions_df["predicted_congestion_level"] = y_pred
            
            predictions_csv_path = os.path.join(processed_dir, "congestion_predictions.csv")
            predictions_df.to_csv(predictions_csv_path, index=False)
            logger.info(f"Saved test predictions to: {predictions_csv_path}")
            
        except Exception as e:
            logger.error(f"Failed to run TCFMu congestion prediction: {str(e)}")

    # Phase 9: Dijkstra Route Optimization (route_opt)
    if args.step in ["all", "route_opt"]:
        logger.info("=== STEP 9: Dijkstra Route Optimization (IM-VRM Final Stage) ===")
        try:
            from src.models.dijkstra_routing import (
                build_weighted_adjacency_graph,
                optimize_greedy_routes
            )
            from src.utils.dijkstra_viz import (
                plot_before_after_comparison,
                plot_grid_edge_cost_heatmap
            )
            
            # Load Greedy baseline routes
            routes_json_path = os.path.join(processed_dir, "baseline_routes.json")
            if not os.path.exists(routes_json_path):
                logger.error(f"Baseline greedy routes not found at {routes_json_path}. Please run --step greedy_route first.")
                return
            with open(routes_json_path, 'r') as f:
                baseline_routes = json.load(f)
                
            # Load congestion predictions
            predictions_csv_path = os.path.join(processed_dir, "congestion_predictions.csv")
            if not os.path.exists(predictions_csv_path):
                logger.error(f"Congestion predictions not found at {predictions_csv_path}. Please run --step tcfmu first.")
                return
            predictions_df = pd.read_csv(predictions_csv_path)
            
            # Load edges csv
            edges_csv_path = os.path.join(processed_dir, "graph_edges.csv")
            edges_df = pd.read_csv(edges_csv_path)
            
            # Load nodes json and parse sorted nodes
            nodes_json_path = os.path.join(processed_dir, "graph_nodes.json")
            with open(nodes_json_path, 'r') as f:
                nodes_dict = json.load(f)
            sorted_nodes = sorted([tuple(map(int, k.split(','))) for k in nodes_dict.keys()])
            node_to_idx = {node: idx for idx, node in enumerate(sorted_nodes)}
            N = len(sorted_nodes)
            
            # Load fleet config
            fleet_json_path = os.path.join(processed_dir, "synthetic_fleet.json")
            if not os.path.exists(fleet_json_path):
                logger.error(f"Synthetic fleet metadata not found at {fleet_json_path}. Please run --step fleet first.")
                return
            with open(fleet_json_path, 'r') as f:
                fleet_data = json.load(f)
                
            # Define weights
            w1, w2, w3 = 0.4, 0.3, 0.3
            logger.info(f"Building weighted adjacency graph with weights w1={w1}, w2={w2}, w3={w3}...")
            
            # Build graph
            weighted_adj_dict, graph_stats = build_weighted_adjacency_graph(
                edges_df, 
                predictions_df, 
                sorted_nodes, 
                config, 
                w1, w2, w3
            )
            
            # Run Dijkstra route re-planning
            logger.info("Optimizing greedy routes using Dijkstra on weighted graph...")
            optimized_routes, comparison_metrics = optimize_greedy_routes(
                baseline_routes,
                weighted_adj_dict,
                edges_df,
                sorted_nodes,
                predictions_df,
                fleet_data
            )
            
            # Save optimized routes to json
            opt_routes_path = os.path.join(processed_dir, "optimized_routes.json")
            with open(opt_routes_path, 'w') as f:
                json.dump(optimized_routes, f, indent=2)
            logger.info(f"Saved optimized routes to: {opt_routes_path}")
            
            # Convert comparison to DataFrame
            comp_df = pd.DataFrame(comparison_metrics)
            
            # Save individual metrics
            dijkstra_metrics_path = os.path.join(outputs_dir, "dijkstra_metrics.csv")
            dijkstra_metrics_df = comp_df[[
                "vehicle_id", "vehicle_type", "depot_id",
                "dijkstra_distance_km", "dijkstra_duration_sec", "dijkstra_avg_congestion",
                "dijkstra_fuel_l", "dijkstra_co2_g"
            ]].rename(columns={
                "dijkstra_distance_km": "distance_km",
                "dijkstra_duration_sec": "duration_sec",
                "dijkstra_avg_congestion": "avg_congestion",
                "dijkstra_fuel_l": "fuel_l",
                "dijkstra_co2_g": "co2_g"
            })
            dijkstra_metrics_df.to_csv(dijkstra_metrics_path, index=False)
            logger.info(f"Saved Dijkstra route metrics to: {dijkstra_metrics_path}")
            
            # Save full comparison report (per-route improvements)
            comparison_csv_path = os.path.join(outputs_dir, "route_comparison.csv")
            comp_df.to_csv(comparison_csv_path, index=False)
            logger.info(f"Saved per-route comparison metrics to: {comparison_csv_path}")
            
            # Aggregate summary statistics
            summary_stats = {
                "weights": {"w1_duration": w1, "w2_distance": w2, "w3_congestion": w3},
                "graph_stats": graph_stats,
                "averages": {
                    "greedy": {
                        "distance_km": float(comp_df["greedy_distance_km"].mean()),
                        "duration_sec": float(comp_df["greedy_duration_sec"].mean()),
                        "avg_congestion": float(comp_df["greedy_avg_congestion"].mean()),
                        "fuel_l": float(comp_df["greedy_fuel_l"].mean()),
                        "co2_g": float(comp_df["greedy_co2_g"].mean())
                    },
                    "dijkstra": {
                        "distance_km": float(comp_df["dijkstra_distance_km"].mean()),
                        "duration_sec": float(comp_df["dijkstra_duration_sec"].mean()),
                        "avg_congestion": float(comp_df["dijkstra_avg_congestion"].mean()),
                        "fuel_l": float(comp_df["dijkstra_fuel_l"].mean()),
                        "co2_g": float(comp_df["dijkstra_co2_g"].mean())
                    },
                    "average_reductions": {
                        "distance_reduction_pct": float(comp_df["reduction_distance_pct"].mean()),
                        "duration_reduction_pct": float(comp_df["reduction_duration_pct"].mean()),
                        "congestion_reduction_pct": float(comp_df["reduction_congestion_pct"].mean()),
                        "fuel_reduction_pct": float(comp_df["reduction_fuel_pct"].mean()),
                        "co2_reduction_pct": float(comp_df["reduction_co2_pct"].mean())
                    }
                }
            }
            
            summary_json_path = os.path.join(outputs_dir, "optimization_summary.json")
            with open(summary_json_path, 'w') as f:
                json.dump(summary_stats, f, indent=2)
            logger.info(f"Saved optimization summary to: {summary_json_path}")
            
            # Print comparative summary
            logger.info("=== Route Optimization Summary (Greedy vs Dijkstra) ===")
            logger.info(f"  Avg Distance:   Greedy={summary_stats['averages']['greedy']['distance_km']:.3f} km vs "
                        f"Dijkstra={summary_stats['averages']['dijkstra']['distance_km']:.3f} km "
                        f"({summary_stats['averages']['average_reductions']['distance_reduction_pct']:.2f}% reduction)")
            logger.info(f"  Avg Duration:   Greedy={summary_stats['averages']['greedy']['duration_sec']:.1f} s vs "
                        f"Dijkstra={summary_stats['averages']['dijkstra']['duration_sec']:.1f} s "
                        f"({summary_stats['averages']['average_reductions']['duration_reduction_pct']:.2f}% reduction)")
            logger.info(f"  Avg Congestion: Greedy={summary_stats['averages']['greedy']['avg_congestion']:.3f} vs "
                        f"Dijkstra={summary_stats['averages']['dijkstra']['avg_congestion']:.3f} "
                        f"({summary_stats['averages']['average_reductions']['congestion_reduction_pct']:.2f}% reduction)")
            logger.info(f"  Avg Fuel:       Greedy={summary_stats['averages']['greedy']['fuel_l']:.3f} L vs "
                        f"Dijkstra={summary_stats['averages']['dijkstra']['fuel_l']:.3f} L "
                        f"({summary_stats['averages']['average_reductions']['fuel_reduction_pct']:.2f}% reduction)")
            logger.info(f"  Avg CO2:        Greedy={summary_stats['averages']['greedy']['co2_g']:.1f} g vs "
                        f"Dijkstra={summary_stats['averages']['dijkstra']['co2_g']:.1f} g "
                        f"({summary_stats['averages']['average_reductions']['co2_reduction_pct']:.2f}% reduction)")
            logger.info("======================================================")
            
            # Visualizations
            # 1. Before/After Comparison Plot (vehicle_001)
            demo_veh = "vehicle_001"
            if demo_veh in optimized_routes and demo_veh in baseline_routes:
                greedy_veh = baseline_routes[demo_veh]
                opt_veh = optimized_routes[demo_veh]
                
                grid_dims = (config["spatial_grid"]["num_rows"], config["spatial_grid"]["num_cols"])
                g_route = [tuple(c) for c in greedy_veh["route_cells"]]
                d_route = [tuple(c) for c in opt_veh["route_cells"]]
                depot = tuple(greedy_veh["mapped_depot_cell"])
                deliveries = [tuple(c) for c in greedy_veh["delivery_cells"]]
                
                compare_plot_path = os.path.join(outputs_dir, "route_comparison_demo.png")
                plot_before_after_comparison(grid_dims, g_route, d_route, depot, deliveries, compare_plot_path)
                logger.info(f"Saved route comparison plot to: {compare_plot_path}")
                
            # 2. Edge-Cost Heatmap Plot
            heatmap_path = os.path.join(outputs_dir, "edge_cost_heatmap.png")
            plot_grid_edge_cost_heatmap(grid_dims, sorted_nodes, weighted_adj_dict, heatmap_path)
            logger.info(f"Saved edge cost heatmap to: {heatmap_path}")
            
        except Exception as e:
            logger.error(f"Failed to run Dijkstra route optimization: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    # Phase 10: Ant Colony Optimization Routing (aco_route)
    if args.step in ["all", "aco_route"]:
        logger.info("=== STEP 10: Ant Colony Optimization Routing (IM-VRM Contribution Stage) ===")
        try:
            import torch
            from src.utils.helpers import haversine_distance
            from src.models.aco_routing import aco_route_planner
            from src.models.dijkstra_routing import (
                build_weighted_adjacency_graph,
                optimize_greedy_routes,
                evaluate_path_metrics,
                load_congestion_ratios,
                load_free_flow_speeds,
                optimize_routes_time_dependent,
                get_predictions_map,
                evaluate_path_metrics_td
            )
            from src.utils.aco_viz import (
                plot_pheromone_heatmap,
                plot_aco_convergence,
                plot_aco_route_comparison
            )
            
            # Load Greedy baseline routes
            greedy_json_path = os.path.join(processed_dir, "baseline_routes.json")
            if not os.path.exists(greedy_json_path):
                logger.error(f"Baseline greedy routes not found at {greedy_json_path}. Please run --step greedy_route first.")
                return
            with open(greedy_json_path, 'r') as f:
                baseline_routes = json.load(f)
                
            # Load Dijkstra optimized routes (Greedy-Dijkstra)
            optimized_json_path = os.path.join(processed_dir, "optimized_routes.json")
            if not os.path.exists(optimized_json_path):
                logger.error(f"Optimized greedy routes not found at {optimized_json_path}. Please run --step route_opt first.")
                return
            with open(optimized_json_path, 'r') as f:
                greedy_dijkstra_routes = json.load(f)
                
            # Load GNN embeddings
            embed_pt_path = os.path.join(processed_dir, "node_embeddings.pt")
            embeddings = torch.load(embed_pt_path)
            
            # Load nodes json and parse sorted nodes
            nodes_json_path = os.path.join(processed_dir, "graph_nodes.json")
            with open(nodes_json_path, 'r') as f:
                nodes_dict = json.load(f)
            sorted_nodes = sorted([tuple(map(int, k.split(','))) for k in nodes_dict.keys()])
            node_to_idx = {node: idx for idx, node in enumerate(sorted_nodes)}
            N = len(sorted_nodes)
            
            # Load edges csv
            edges_csv_path = os.path.join(processed_dir, "graph_edges.csv")
            edges_df = pd.read_csv(edges_csv_path)
            
            # Load congestion predictions
            predictions_csv_path = os.path.join(processed_dir, "congestion_predictions.csv")
            predictions_df = pd.read_csv(predictions_csv_path)
            
            # Load fleet config
            fleet_json_path = os.path.join(processed_dir, "synthetic_fleet.json")
            with open(fleet_json_path, 'r') as f:
                fleet_data = json.load(f)
                
            # Map fleet configs
            fleet_configs = {v["vehicle_id"]: v["config"] for v in fleet_data["fleet"]}
            
            # Build transition adjacency dictionary with (neighbor_idx, duration, distance)
            bbox = config["preprocessing"]["bbox"]
            grid_dims = (config["spatial_grid"]["num_rows"], config["spatial_grid"]["num_cols"])
            min_lat, min_lon, max_lat, max_lon = bbox
            num_rows, num_cols = grid_dims
            lat_step = (max_lat - min_lat) / num_rows
            lon_step = (max_lon - min_lon) / num_cols
            
            adj_dict = {}
            for _, edge in edges_df.iterrows():
                u = (int(edge["grid_row"]), int(edge["grid_col"]))
                v = (int(edge["next_row"]), int(edge["next_col"]))
                if u in node_to_idx and v in node_to_idx:
                    u_idx = node_to_idx[u]
                    v_idx = node_to_idx[v]
                    
                    # Compute geodetic distance between centroids
                    u_lat = min_lat + (u[0] + 0.5) * lat_step
                    u_lon = min_lon + (u[1] + 0.5) * lon_step
                    v_lat = min_lat + (v[0] + 0.5) * lat_step
                    v_lon = min_lon + (v[1] + 0.5) * lon_step
                    dist_m = haversine_distance(u_lat, u_lon, v_lat, v_lon)
                    
                    duration = float(edge["avg_duration_sec"])
                    
                    if u_idx not in adj_dict:
                        adj_dict[u_idx] = []
                    adj_dict[u_idx].append((v_idx, duration, dist_m))
            
            # Configure ACO parameters
            aco_params = {
                "ants": 15,
                "iterations": 25,
                "alpha": 1.0,           # pheromone weight
                "beta": 2.0,             # heuristic weight
                "rho_global": 0.2,       # global pheromone decay
                "rho_local": 0.1,        # local pheromone decay
                "q0": 0.7,               # exploitation/exploration balance
                "Q": 1.0,                # reinforcement scale
                "alpha1": 1.0,           # embedding coef
                "alpha2": 1.5,           # congestion coef
                "alpha3": 1.0,           # duration coef
                "alpha4": 1.0,           # distance coef
                
                # Objectives weights
                "w_distance": 0.2,
                "w_duration": 0.2,
                "w_congestion": 0.2,
                "w_fuel": 0.2,
                "w_co2": 0.2,
                "w_penalty": 100.0       # heavy penalty for unreachable delivery targets
            }
            
            logger.info("Executing Ant Colony Optimization routing...")
            aco_routes = {}
            aco_metrics_list = []
            aco_decision_traces = {}
            
            # Store pheromone history and node labels of vehicle_001 for diagnostics
            demo_pheromone_history = None
            demo_node_labels = None
            
            for vehicle in fleet_data["fleet"]:
                v_id = vehicle["vehicle_id"]
                greedy_veh = baseline_routes.get(v_id)
                if greedy_veh is None:
                    continue
                    
                depot_cell = tuple(greedy_veh["mapped_depot_cell"])
                depot_idx = node_to_idx[depot_cell]
                delivery_cells = [tuple(c) for c in greedy_veh["delivery_cells"]]
                delivery_indices = [node_to_idx[c] for c in delivery_cells]
                
                vehicle_config = fleet_configs.get(v_id)
                
                # Fetch Greedy baseline metrics for this vehicle to normalize objectives
                greedy_nodes = greedy_veh["route_cells"]
                greedy_path_nodes = [node_to_idx[tuple(c)] for c in greedy_nodes if tuple(c) in node_to_idx]
                greedy_metrics = evaluate_path_metrics(greedy_path_nodes, edges_df, sorted_nodes, predictions_df, vehicle_config)
                
                # Plan route using ACO
                aco_plan = aco_route_planner(
                    depot_idx,
                    delivery_indices,
                    embeddings,
                    adj_dict,
                    N,
                    predictions_df,
                    sorted_nodes,
                    edges_df,
                    vehicle_config,
                    greedy_metrics,
                    aco_params
                )
                
                # Store
                aco_routes[v_id] = {
                    "vehicle_id": v_id,
                    "type": greedy_veh["type"],
                    "depot_id": greedy_veh["depot_id"],
                    "depot_cell": greedy_veh["depot_cell"],
                    "mapped_depot_cell": greedy_veh["mapped_depot_cell"],
                    "delivery_cells": greedy_veh["delivery_cells"],
                    "route_cells": aco_plan["route_cells"],
                    "decision_trace": aco_plan["decision_trace"],
                    "unvisited_remaining": aco_plan["unvisited_remaining"],
                    "total_distance_km": aco_plan["total_distance_km"],
                    "total_duration_sec": aco_plan["total_duration_sec"],
                    "avg_congestion": aco_plan["avg_congestion"],
                    "fuel_l": aco_plan["fuel_l"],
                    "co2_g": aco_plan["co2_g"],
                    "objective_value": aco_plan["objective_value"]
                }
                
                aco_metrics_list.append({
                    "vehicle_id": v_id,
                    "vehicle_type": greedy_veh["type"],
                    "depot_id": greedy_veh["depot_id"],
                    
                    # Greedy baseline reference metrics
                    "greedy_distance_km": greedy_metrics["distance_km"],
                    "greedy_duration_sec": greedy_metrics["duration_sec"],
                    "greedy_avg_congestion": greedy_metrics["avg_congestion"],
                    "greedy_fuel_l": greedy_metrics["fuel_l"],
                    "greedy_co2_g": greedy_metrics["co2_g"],
                    
                    # ACO baseline metrics
                    "aco_distance_km": aco_plan["total_distance_km"],
                    "aco_duration_sec": aco_plan["total_duration_sec"],
                    "aco_avg_congestion": aco_plan["avg_congestion"],
                    "aco_fuel_l": aco_plan["fuel_l"],
                    "aco_co2_g": aco_plan["co2_g"],
                    "aco_objective_value": aco_plan["objective_value"]
                })
                
                aco_decision_traces[v_id] = {
                    "convergence_history": aco_plan["convergence_history"],
                    "unvisited_remaining_count": len(aco_plan["unvisited_remaining"])
                }
                
                if v_id == "vehicle_001":
                    demo_pheromone_history = aco_plan["pheromone_history"]
                    demo_node_labels = aco_plan["node_labels"]
                    
            # Save ACO baseline routes JSON
            aco_routes_path = os.path.join(processed_dir, "aco_routes.json")
            with open(aco_routes_path, 'w') as f:
                json.dump(aco_routes, f, indent=2)
            logger.info(f"Saved ACO baseline routes to: {aco_routes_path}")
            
            # Save ACO decision traces JSON
            aco_traces_path = os.path.join(outputs_dir, "aco_decision_traces.json")
            with open(aco_traces_path, 'w') as f:
                json.dump(aco_decision_traces, f, indent=2)
            logger.info(f"Saved ACO convergence traces to: {aco_traces_path}")
            
            # Save pheromone history of vehicle_001 JSON
            pheromone_history_path = os.path.join(outputs_dir, "pheromone_history.json")
            with open(pheromone_history_path, 'w') as f:
                json.dump({
                    "vehicle_id": "vehicle_001",
                    "node_labels": demo_node_labels,
                    "pheromone_history": demo_pheromone_history
                }, f, indent=2)
            logger.info(f"Saved vehicle_001 pheromone history to: {pheromone_history_path}")
            
            # ----------------------------------------------------
            # DOWNSTREAM DIJKSTRA RE-OPTIMIZATION ON ACO ROUTES
            # ----------------------------------------------------
            logger.info("Executing Dijkstra route optimization on ACO routes...")
            # Reuse same weights: w1=0.4, w2=0.3, w3=0.3
            w1, w2, w3 = 0.4, 0.3, 0.3
            weighted_adj_dict, normalization_stats = build_weighted_adjacency_graph(
                edges_df, 
                predictions_df, 
                sorted_nodes, 
                config, 
                w1, w2, w3
            )
            normalization_bounds = normalization_stats["normalization_bounds"]
            
            aco_optimized_routes, aco_opt_metrics = optimize_greedy_routes(
                aco_routes,
                weighted_adj_dict,
                edges_df,
                sorted_nodes,
                predictions_df,
                fleet_data
            )
            
            # Save Dijkstra-optimized ACO routes JSON
            aco_opt_routes_path = os.path.join(processed_dir, "aco_optimized_routes.json")
            with open(aco_opt_routes_path, 'w') as f:
                json.dump(aco_optimized_routes, f, indent=2)
            logger.info(f"Saved Dijkstra-optimized ACO routes to: {aco_opt_routes_path}")
            
            # ----------------------------------------------------
            # TIME-DEPENDENT ROUTING EVALUATION FOR MULTIPLE SHIFTS
            # ----------------------------------------------------
            logger.info("Starting Time-Dependent Context-Aware Dijkstra Routing evaluation...")
            global_ratios, cell_ratios = load_congestion_ratios(processed_dir)
            free_flow_speeds = load_free_flow_speeds(processed_dir)
            predictions_map = get_predictions_map(predictions_df)
            
            # Load Greedy + Static Dijkstra routes from processed_dir
            greedy_static_path = os.path.join(processed_dir, "optimized_routes.json")
            if not os.path.exists(greedy_static_path):
                logger.error(f"Greedy static routes not found at {greedy_static_path}. Run --step route_opt first.")
                return
            with open(greedy_static_path, 'r') as f:
                greedy_static_routes = json.load(f)

            # Node mapping helper
            node_to_idx = {node: idx for idx, node in enumerate(sorted_nodes)}
            fleet_configs = {v["vehicle_id"]: v["config"] for v in fleet_data["fleet"]}

            td_results = {}
            td_metrics_records = []
            
            start_hours = [8, 12, 17, 20]
            
            for h in start_hours:
                start_time_sec = h * 3600
                logger.info(f"Evaluating shift starting at {h:02d}:00...")
                
                # 1. Greedy + Static Dijkstra routes evaluated under TD traffic of hour H
                greedy_static_metrics = []
                for v_id, route in greedy_static_routes.items():
                    v_config = fleet_configs.get(v_id)
                    path_nodes = [node_to_idx[tuple(c)] for c in route["route_cells"] if tuple(c) in node_to_idx]
                    metrics = evaluate_path_metrics_td(
                        path_nodes, edges_df, sorted_nodes, predictions_map,
                        free_flow_speeds, global_ratios, cell_ratios, v_config, start_time_sec
                    )
                    metrics["vehicle_id"] = v_id
                    greedy_static_metrics.append(metrics)
                df_greedy_static = pd.DataFrame(greedy_static_metrics)

                # 2. Greedy + Time-Dependent Dijkstra routes evaluated under TD traffic of hour H
                greedy_td_routes, greedy_td_metrics = optimize_routes_time_dependent(
                    baseline_routes,
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
                )
                df_greedy_td = pd.DataFrame(greedy_td_metrics)
                
                # 3. ACO + Static Dijkstra routes evaluated under TD traffic of hour H
                aco_static_metrics = []
                for v_id, route in aco_optimized_routes.items():
                    v_config = fleet_configs.get(v_id)
                    path_nodes = [node_to_idx[tuple(c)] for c in route["route_cells"] if tuple(c) in node_to_idx]
                    metrics = evaluate_path_metrics_td(
                        path_nodes, edges_df, sorted_nodes, predictions_map,
                        free_flow_speeds, global_ratios, cell_ratios, v_config, start_time_sec
                    )
                    metrics["vehicle_id"] = v_id
                    aco_static_metrics.append(metrics)
                df_aco_static = pd.DataFrame(aco_static_metrics)

                # 4. ACO + Time-Dependent Dijkstra routes evaluated under TD traffic of hour H
                aco_td_routes, aco_td_metrics = optimize_routes_time_dependent(
                    aco_routes,
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
                )
                df_aco_td = pd.DataFrame(aco_td_metrics)
                
                # Save routes for this hour
                hour_routes_path = os.path.join(processed_dir, f"td_dijkstra_routes_hour{h}.json")
                with open(hour_routes_path, 'w') as f:
                    json.dump({
                        "greedy_static_routes": greedy_static_routes,
                        "greedy_td_routes": greedy_td_routes,
                        "aco_static_routes": aco_optimized_routes,
                        "aco_td_routes": aco_td_routes
                    }, f, indent=2)
                
                # Gather comparison stats for this shift
                # Merge all four dataframes for detailed vehicle comparison CSV
                merged_td = pd.merge(
                    df_greedy_static[["vehicle_id", "distance_km", "duration_sec", "fuel_l", "co2_g", "avg_congestion"]].rename(columns=lambda x: f"greedy_static_{x}" if x != "vehicle_id" else x),
                    df_greedy_td[["vehicle_id", "dijkstra_distance_km", "dijkstra_duration_sec", "dijkstra_fuel_l", "dijkstra_co2_g", "dijkstra_avg_congestion"]].rename(columns=lambda x: x.replace("dijkstra", "greedy_td") if x != "vehicle_id" else x),
                    on="vehicle_id"
                )
                merged_td = pd.merge(
                    merged_td,
                    df_aco_static[["vehicle_id", "distance_km", "duration_sec", "fuel_l", "co2_g", "avg_congestion"]].rename(columns=lambda x: f"aco_static_{x}" if x != "vehicle_id" else x),
                    on="vehicle_id"
                )
                merged_td = pd.merge(
                    merged_td,
                    df_aco_td[["vehicle_id", "dijkstra_distance_km", "dijkstra_duration_sec", "dijkstra_fuel_l", "dijkstra_co2_g", "dijkstra_avg_congestion"]].rename(columns=lambda x: x.replace("dijkstra", "aco_td") if x != "vehicle_id" else x),
                    on="vehicle_id"
                )
                merged_td["start_hour"] = h
                td_metrics_records.append(merged_td)
                
                td_results[h] = {
                    "greedy_static": {
                        "distance_km": float(df_greedy_static["distance_km"].mean()),
                        "duration_sec": float(df_greedy_static["duration_sec"].mean()),
                        "fuel_l": float(df_greedy_static["fuel_l"].mean()),
                        "co2_g": float(df_greedy_static["co2_g"].mean()),
                        "avg_congestion": float(df_greedy_static["avg_congestion"].mean())
                    },
                    "greedy_td": {
                        "distance_km": float(df_greedy_td["dijkstra_distance_km"].mean()),
                        "duration_sec": float(df_greedy_td["dijkstra_duration_sec"].mean()),
                        "fuel_l": float(df_greedy_td["dijkstra_fuel_l"].mean()),
                        "co2_g": float(df_greedy_td["dijkstra_co2_g"].mean()),
                        "avg_congestion": float(df_greedy_td["dijkstra_avg_congestion"].mean())
                    },
                    "aco_static": {
                        "distance_km": float(df_aco_static["distance_km"].mean()),
                        "duration_sec": float(df_aco_static["duration_sec"].mean()),
                        "fuel_l": float(df_aco_static["fuel_l"].mean()),
                        "co2_g": float(df_aco_static["co2_g"].mean()),
                        "avg_congestion": float(df_aco_static["avg_congestion"].mean())
                    },
                    "aco_td": {
                        "distance_km": float(df_aco_td["dijkstra_distance_km"].mean()),
                        "duration_sec": float(df_aco_td["dijkstra_duration_sec"].mean()),
                        "fuel_l": float(df_aco_td["dijkstra_fuel_l"].mean()),
                        "co2_g": float(df_aco_td["dijkstra_co2_g"].mean()),
                        "avg_congestion": float(df_aco_td["dijkstra_avg_congestion"].mean())
                    }
                }
            
            # Combine and save TD comparative metrics CSV
            td_metrics_all_df = pd.concat(td_metrics_records, ignore_index=True)
            td_metrics_csv_path = os.path.join(outputs_dir, "td_aco_metrics.csv")
            td_metrics_all_df.to_csv(td_metrics_csv_path, index=False)
            logger.info(f"Saved Time-Dependent metrics comparisons to: {td_metrics_csv_path}")

            # Construct consolidated metrics comparisons DataFrame
            aco_base_df = pd.DataFrame(aco_metrics_list)
            aco_opt_df = pd.DataFrame(aco_opt_metrics)
            
            # Filter optimized metrics to keep only Dijkstra-optimized columns to avoid duplicate collisions
            aco_opt_df_clean = aco_opt_df[[
                "vehicle_id", "dijkstra_distance_km", "dijkstra_duration_sec", 
                "dijkstra_avg_congestion", "dijkstra_fuel_l", "dijkstra_co2_g"
            ]]
            
            # Merge baseline and optimized comparisons
            merged_metrics = pd.merge(
                aco_base_df, 
                aco_opt_df_clean, 
                on="vehicle_id"
            )
            
            # Rename columns to provide clear comparison fields
            greedy_dijkstra_metrics = []
            for v_id, gd_route in greedy_dijkstra_routes.items():
                greedy_dijkstra_metrics.append({
                    "vehicle_id": v_id,
                    "greedy_dijkstra_distance_km": gd_route["distance_km"],
                    "greedy_dijkstra_duration_sec": gd_route["duration_sec"],
                    "greedy_dijkstra_avg_congestion": gd_route["avg_congestion"],
                    "greedy_dijkstra_fuel_l": gd_route["fuel_l"],
                    "greedy_dijkstra_co2_g": gd_route["co2_g"]
                })
            gd_df = pd.DataFrame(greedy_dijkstra_metrics)
            
            final_comp_df = pd.merge(merged_metrics, gd_df, on="vehicle_id")
            
            # Compute improvement columns: Greedy-Dijkstra vs ACO-Dijkstra
            final_comp_df["improvement_distance_km"] = final_comp_df["greedy_dijkstra_distance_km"] - final_comp_df["dijkstra_distance_km"]
            final_comp_df["improvement_duration_sec"] = final_comp_df["greedy_dijkstra_duration_sec"] - final_comp_df["dijkstra_duration_sec"]
            final_comp_df["improvement_congestion"] = final_comp_df["greedy_dijkstra_avg_congestion"] - final_comp_df["dijkstra_avg_congestion"]
            final_comp_df["improvement_fuel_l"] = final_comp_df["greedy_dijkstra_fuel_l"] - final_comp_df["dijkstra_fuel_l"]
            final_comp_df["improvement_co2_g"] = final_comp_df["greedy_dijkstra_co2_g"] - final_comp_df["dijkstra_co2_g"]
            
            def pct_red(diff, base):
                return (diff / base * 100) if base > 0 else 0.0
                
            final_comp_df["reduction_distance_pct"] = final_comp_df.apply(lambda r: pct_red(r["improvement_distance_km"], r["greedy_dijkstra_distance_km"]), axis=1)
            final_comp_df["reduction_duration_pct"] = final_comp_df.apply(lambda r: pct_red(r["improvement_duration_sec"], r["greedy_dijkstra_duration_sec"]), axis=1)
            final_comp_df["reduction_congestion_pct"] = final_comp_df.apply(lambda r: pct_red(r["improvement_congestion"], r["greedy_dijkstra_avg_congestion"]), axis=1)
            final_comp_df["reduction_fuel_pct"] = final_comp_df.apply(lambda r: pct_red(r["improvement_fuel_l"], r["greedy_dijkstra_fuel_l"]), axis=1)
            final_comp_df["reduction_co2_pct"] = final_comp_df.apply(lambda r: pct_red(r["improvement_co2_g"], r["greedy_dijkstra_co2_g"]), axis=1)
            
            # Save ACO metrics CSV
            aco_metrics_path = os.path.join(outputs_dir, "aco_metrics.csv")
            final_comp_df[[
                "vehicle_id", "vehicle_type", "depot_id",
                "aco_distance_km", "aco_duration_sec", "aco_avg_congestion", "aco_fuel_l", "aco_co2_g",
                "dijkstra_distance_km", "dijkstra_duration_sec", "dijkstra_avg_congestion", "dijkstra_fuel_l", "dijkstra_co2_g",
                "improvement_distance_km", "improvement_duration_sec", "improvement_congestion", "improvement_fuel_l", "improvement_co2_g",
                "reduction_distance_pct", "reduction_duration_pct", "reduction_congestion_pct", "reduction_fuel_pct", "reduction_co2_pct"
            ]].to_csv(aco_metrics_path, index=False)
            logger.info(f"Saved ACO route metrics to: {aco_metrics_path}")
            
            # Save consolidated comparative evaluation reports
            summary_stats = {
                "aco_params": aco_params,
                "congestion_ratios_derived": {
                    "global": global_ratios,
                    "cell_specific_count": len(cell_ratios)
                },
                "td_sensitivity_analysis": td_results,
                "averages": {
                    "greedy_baseline": {
                        "distance_km": float(final_comp_df["greedy_distance_km"].mean()),
                        "duration_sec": float(final_comp_df["greedy_duration_sec"].mean()),
                        "avg_congestion": float(final_comp_df["greedy_avg_congestion"].mean()),
                        "fuel_l": float(final_comp_df["greedy_fuel_l"].mean()),
                        "co2_g": float(final_comp_df["greedy_co2_g"].mean())
                    },
                    "aco_baseline": {
                        "distance_km": float(final_comp_df["aco_distance_km"].mean()),
                        "duration_sec": float(final_comp_df["aco_duration_sec"].mean()),
                        "avg_congestion": float(final_comp_df["aco_avg_congestion"].mean()),
                        "fuel_l": float(final_comp_df["aco_fuel_l"].mean()),
                        "co2_g": float(final_comp_df["aco_co2_g"].mean())
                    },
                    "greedy_dijkstra_optimized": {
                        "distance_km": float(final_comp_df["greedy_dijkstra_distance_km"].mean()),
                        "duration_sec": float(final_comp_df["greedy_dijkstra_duration_sec"].mean()),
                        "avg_congestion": float(final_comp_df["greedy_dijkstra_avg_congestion"].mean()),
                        "fuel_l": float(final_comp_df["greedy_dijkstra_fuel_l"].mean()),
                        "co2_g": float(final_comp_df["greedy_dijkstra_co2_g"].mean())
                    },
                    "aco_dijkstra_optimized": {
                        "distance_km": float(final_comp_df["dijkstra_distance_km"].mean()),
                        "duration_sec": float(final_comp_df["dijkstra_duration_sec"].mean()),
                        "avg_congestion": float(final_comp_df["dijkstra_avg_congestion"].mean()),
                        "fuel_l": float(final_comp_df["dijkstra_fuel_l"].mean()),
                        "co2_g": float(final_comp_df["dijkstra_co2_g"].mean())
                    },
                    "optimized_improvements_pct": {
                        "distance_reduction_pct": float(final_comp_df["reduction_distance_pct"].mean()),
                        "duration_reduction_pct": float(final_comp_df["reduction_duration_pct"].mean()),
                        "congestion_reduction_pct": float(final_comp_df["reduction_congestion_pct"].mean()),
                        "fuel_reduction_pct": float(final_comp_df["reduction_fuel_pct"].mean()),
                        "co2_reduction_pct": float(final_comp_df["reduction_co2_pct"].mean())
                    }
                }
            }
            
            aco_summary_path = os.path.join(outputs_dir, "aco_summary.json")
            with open(aco_summary_path, 'w') as f:
                json.dump(summary_stats, f, indent=2)
            logger.info(f"Saved ACO comparative summary to: {aco_summary_path}")
            
            # Print Comparative Table
            logger.info("=== Comparative Route Performance (ACO-Dijkstra vs Greedy-Dijkstra) ===")
            logger.info(f"  Distance (km): Greedy-Opt={summary_stats['averages']['greedy_dijkstra_optimized']['distance_km']:.3f} "
                        f"vs ACO-Opt={summary_stats['averages']['aco_dijkstra_optimized']['distance_km']:.3f} "
                        f"({summary_stats['averages']['optimized_improvements_pct']['distance_reduction_pct']:.2f}% reduction)")
            logger.info(f"  Duration (s):  Greedy-Opt={summary_stats['averages']['greedy_dijkstra_optimized']['duration_sec']:.1f} "
                        f"vs ACO-Opt={summary_stats['averages']['aco_dijkstra_optimized']['duration_sec']:.1f} "
                        f"({summary_stats['averages']['optimized_improvements_pct']['duration_reduction_pct']:.2f}% reduction)")
            logger.info(f"  Congestion:    Greedy-Opt={summary_stats['averages']['greedy_dijkstra_optimized']['avg_congestion']:.3f} "
                        f"vs ACO-Opt={summary_stats['averages']['aco_dijkstra_optimized']['avg_congestion']:.3f} "
                        f"({summary_stats['averages']['optimized_improvements_pct']['congestion_reduction_pct']:.2f}% reduction)")
            logger.info(f"  Fuel (L):      Greedy-Opt={summary_stats['averages']['greedy_dijkstra_optimized']['fuel_l']:.3f} "
                        f"vs ACO-Opt={summary_stats['averages']['aco_dijkstra_optimized']['fuel_l']:.3f} "
                        f"({summary_stats['averages']['optimized_improvements_pct']['fuel_reduction_pct']:.2f}% reduction)")
            logger.info(f"  CO2 (g):       Greedy-Opt={summary_stats['averages']['greedy_dijkstra_optimized']['co2_g']:.1f} "
                        f"vs ACO-Opt={summary_stats['averages']['aco_dijkstra_optimized']['co2_g']:.1f} "
                        f"({summary_stats['averages']['optimized_improvements_pct']['co2_reduction_pct']:.2f}% reduction)")
            logger.info("=======================================================================")
            
            logger.info("=== Time-Dependent Routing Sensitivity Analysis (Full Factorial Comparison) ===")
            logger.info("Hour  | Model         | Avg Dist (km) | Avg Dur (sec) | Avg Fuel (L) | Avg CO2 (g) | Avg Congestion")
            logger.info("------|---------------|---------------|---------------|--------------|-------------|---------------")
            for h in start_hours:
                res = td_results[h]
                logger.info(f"{h:02d}:00 | Greedy-Static | {res['greedy_static']['distance_km']:13.3f} | {res['greedy_static']['duration_sec']:13.1f} | {res['greedy_static']['fuel_l']:12.3f} | {res['greedy_static']['co2_g']:11.1f} | {res['greedy_static']['avg_congestion']:14.3f}")
                logger.info(f"{h:02d}:00 | Greedy-TD     | {res['greedy_td']['distance_km']:13.3f} | {res['greedy_td']['duration_sec']:13.1f} | {res['greedy_td']['fuel_l']:12.3f} | {res['greedy_td']['co2_g']:11.1f} | {res['greedy_td']['avg_congestion']:14.3f}")
                logger.info(f"{h:02d}:00 | ACO-Static    | {res['aco_static']['distance_km']:13.3f} | {res['aco_static']['duration_sec']:13.1f} | {res['aco_static']['fuel_l']:12.3f} | {res['aco_static']['co2_g']:11.1f} | {res['aco_static']['avg_congestion']:14.3f}")
                logger.info(f"{h:02d}:00 | ACO-TD        | {res['aco_td']['distance_km']:13.3f} | {res['aco_td']['duration_sec']:13.1f} | {res['aco_td']['fuel_l']:12.3f} | {res['aco_td']['co2_g']:11.1f} | {res['aco_td']['avg_congestion']:14.3f}")
                logger.info("------|---------------|---------------|---------------|--------------|-------------|---------------")
            logger.info("=====================================================================================================")

            
            # Visualizations
            # 1. Pheromone Heatmap for vehicle_001
            heatmap_path = os.path.join(outputs_dir, "pheromone_heatmap_demo.png")
            plot_pheromone_heatmap(demo_pheromone_history[-1], demo_node_labels, heatmap_path)
            logger.info(f"Saved pheromone heatmap to: {heatmap_path}")
            
            # 2. Convergence Curve for vehicle_001
            convergence_path = os.path.join(outputs_dir, "aco_convergence_demo.png")
            plot_aco_convergence(aco_decision_traces["vehicle_001"]["convergence_history"], convergence_path)
            logger.info(f"Saved ACO convergence curve to: {convergence_path}")
            
            # 3. Route Comparison Plot (Greedy vs ACO)
            greedy_veh = baseline_routes.get("vehicle_001")
            aco_veh = aco_routes.get("vehicle_001")
            if greedy_veh and aco_veh:
                g_route = [tuple(c) for c in greedy_veh["route_cells"]]
                a_route = [tuple(c) for c in aco_veh["route_cells"]]
                depot = tuple(greedy_veh["mapped_depot_cell"])
                deliveries = [tuple(c) for c in greedy_veh["delivery_cells"]]
                
                route_comp_path = os.path.join(outputs_dir, "aco_route_comparison_demo.png")
                plot_aco_route_comparison(grid_dims, g_route, a_route, depot, deliveries, route_comp_path)
                logger.info(f"Saved Greedy vs ACO route comparison demo plot to: {route_comp_path}")
                
        except Exception as e:
            logger.error(f"Failed to run ACO route selection: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    logger.info("IM-VRM baseline pipeline completed successfully.")

if __name__ == "__main__":
    main()

