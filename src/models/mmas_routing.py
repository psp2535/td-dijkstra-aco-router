import numpy as np
import concurrent.futures
from src.utils.helpers import haversine_distance
from src.models.dijkstra_routing import dijkstra_time_dependent_pathfinder

def parallel_mmas_route_planner(depot_node_idx, delivery_node_indices, embeddings, adj_dict, N, sorted_nodes, edges_df, 
                                aco_config=None, weighted_adj_dict=None, bbox=None, grid_dims=None,
                                predictions_map=None, free_flow_speeds=None, global_ratios=None, 
                                cell_ratios=None, normalization_bounds=None):
    """
    Parallel Max-Min Ant System (MMAS) for real-time routing.
    Computes an exact A* cost matrix to connect the 10 depots perfectly.
    """
    if aco_config is None:
        aco_config = {}
        
    num_ants = aco_config.get("ants", 15)
    max_iterations = aco_config.get("iterations", 10)
    alpha = aco_config.get("alpha", 1.0)
    beta = aco_config.get("beta", 3.0)
    rho = aco_config.get("rho", 0.1)
    q0 = aco_config.get("q0", 0.9)
    Tau_max = aco_config.get("Tau_max", 5.0)
    Tau_min = aco_config.get("Tau_min", 0.01)
    
    aco_nodes = [depot_node_idx] + list(delivery_node_indices)
    K = len(aco_nodes)
    
    # 1. Precompute Exact A* Cost Matrix
    heuristic_matrix = np.zeros((K, K))
    
    for i in range(K):
        for j in range(K):
            if i != j:
                u = aco_nodes[i]
                v = aco_nodes[j]
                
                # Use TD Dijkstra to find the true road network cost between depots
                if weighted_adj_dict and predictions_map is not None:
                    _, cost, _, _ = dijkstra_time_dependent_pathfinder(
                        u, v, 8 * 3600.0, weighted_adj_dict, N,
                        predictions_map, sorted_nodes, free_flow_speeds,
                        global_ratios, cell_ratios, normalization_bounds,
                        0.33, 0.33, 0.34
                    )
                    heuristic_matrix[i, j] = 1.0 / (cost + 1e-6)
                else:
                    # Fallback to Haversine if A* data is missing
                    lat_u, lon_u = sorted_nodes[u]
                    lat_v, lon_v = sorted_nodes[v]
                    d = haversine_distance(lat_u, lon_u, lat_v, lon_v)
                    heuristic_matrix[i, j] = 1.0 / (d + 1e-6)
                
    Tau = np.full((K, K), Tau_max, dtype=float)
    
    best_overall_seq = None
    best_overall_dist = float('inf')
    
    rng = np.random.default_rng()
    
    def simulate_ant(start_pos):
        unvisited = set(range(1, K))
        curr_pos = start_pos
        seq = [curr_pos]
        dist = 0.0
        
        while unvisited:
            probs = []
            cands = list(unvisited)
            for cand in cands:
                tau = Tau[curr_pos, cand]
                eta = heuristic_matrix[curr_pos, cand]
                probs.append((tau ** alpha) * (eta ** beta))
                
            probs = np.array(probs)
            if probs.sum() == 0:
                break
                
            if rng.uniform(0, 1) < q0:
                next_pos = cands[np.argmax(probs)]
            else:
                probs = probs / probs.sum()
                next_pos = rng.choice(cands, p=probs)
                
            seq.append(next_pos)
            dist += 1.0 / heuristic_matrix[curr_pos, next_pos]
            unvisited.remove(next_pos)
            curr_pos = next_pos
            
        # Return to depot
        seq.append(start_pos)
        dist += 1.0 / heuristic_matrix[curr_pos, start_pos]
        return seq, dist

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_ants) as executor:
        for _ in range(max_iterations):
            futures = [executor.submit(simulate_ant, 0) for _ in range(num_ants)]
            iteration_best_seq = None
            iteration_best_dist = float('inf')
            
            for f in concurrent.futures.as_completed(futures):
                seq, dist = f.result()
                if dist < iteration_best_dist:
                    iteration_best_dist = dist
                    iteration_best_seq = seq
                    
            if iteration_best_dist < best_overall_dist:
                best_overall_dist = iteration_best_dist
                best_overall_seq = iteration_best_seq
                
            # MMAS Pheromone Update
            Tau = (1.0 - rho) * Tau
            if iteration_best_seq:
                delta = 1.0 / iteration_best_dist
                for i in range(len(iteration_best_seq) - 1):
                    u, v = iteration_best_seq[i], iteration_best_seq[i+1]
                    Tau[u, v] += delta
            
            Tau = np.clip(Tau, Tau_min, Tau_max)

    if not best_overall_seq:
        return {"route_nodes": [], "target_seq": []}
        
    final_target_seq = [aco_nodes[p] for p in best_overall_seq]
    
    return {
        "target_seq": final_target_seq,
        "heuristic_dist": best_overall_dist
    }
