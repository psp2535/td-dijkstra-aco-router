import matplotlib.pyplot as plt
import numpy as np

def plot_before_after_comparison(grid_dims, greedy_route, dijkstra_route, depot, deliveries, save_path):
    """
    Plot and compare the Greedy route (before) vs Dijkstra route (after) on the 2D spatial grid.
    """
    num_rows, num_cols = grid_dims
    grid_map = np.zeros((num_rows, num_cols))
    
    fig, ax = plt.subplots(figsize=(9, 9))
    
    # Background grid ticks
    ax.imshow(grid_map, origin='lower', cmap='binary', extent=[0, num_cols, 0, num_rows], alpha=0.03)
    ax.set_xticks(np.arange(0, num_cols + 1, 5))
    ax.set_yticks(np.arange(0, num_rows + 1, 5))
    ax.grid(True, which='both', color='gray', linestyle='--', alpha=0.2)
    
    # 1. Plot Greedy baseline route (dotted gray/orange path)
    if len(greedy_route) > 1:
        greedy_arr = np.array(greedy_route)
        ax.plot(
            greedy_arr[:, 1] + 0.5,
            greedy_arr[:, 0] + 0.5,
            color='gray',
            linestyle='--',
            linewidth=2.0,
            alpha=0.6,
            label='Greedy Route (Before)',
            zorder=2
        )
        
    # 2. Plot Dijkstra-optimized route (solid vibrant blue path)
    if len(dijkstra_route) > 1:
        dijkstra_arr = np.array(dijkstra_route)
        ax.plot(
            dijkstra_arr[:, 1] + 0.5,
            dijkstra_arr[:, 0] + 0.5,
            color='dodgerblue',
            linewidth=3.0,
            label='Dijkstra Green Route (After)',
            zorder=3
        )
        
        # Path transition arrows for Dijkstra Route
        for i in range(len(dijkstra_route) - 1):
            start = dijkstra_route[i]
            end = dijkstra_route[i+1]
            dx = end[1] - start[1]
            dy = end[0] - start[0]
            if dx != 0 or dy != 0:
                ax.annotate(
                    "",
                    xy=(end[1] + 0.5, end[0] + 0.5),
                    xytext=(start[1] + 0.5, start[0] + 0.5),
                    arrowprops=dict(arrowstyle="->", color="dodgerblue", lw=1.5, alpha=0.8),
                    zorder=4
                )
                
    # 3. Plot delivery locations
    deliv_arr = np.array(deliveries)
    if len(deliv_arr) > 0:
        ax.scatter(
            deliv_arr[:, 1] + 0.5,
            deliv_arr[:, 0] + 0.5,
            color='forestgreen',
            s=130,
            marker='o',
            edgecolors='black',
            label='Delivery Nodes',
            zorder=5
        )
        
    # 4. Plot depot
    ax.scatter(
        depot[1] + 0.5,
        depot[0] + 0.5,
        color='crimson',
        s=250,
        marker='*',
        edgecolors='black',
        label='Vehicle Depot',
        zorder=6
    )
    
    ax.set_xlim(0, num_cols)
    ax.set_ylim(0, num_rows)
    ax.set_xlabel("Grid Column")
    ax.set_ylabel("Grid Row")
    ax.set_title("IM-VRM Routing Path Optimization Comparison", fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.95)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_edge_cost_heatmap(grid_dims, edges_df, weighted_adj_dict, save_path):
    """
    Plot a spatial heatmap of normalized edge costs to visualize high-cost regions.
    """
    num_rows, num_cols = grid_dims
    heatmap = np.zeros((num_rows, num_cols))
    counts = np.zeros((num_rows, num_cols))
    
    # Calculate average outgoing edge cost for each cell
    for u_idx, neighbors in weighted_adj_dict.items():
        # Retrieve row, col coordinate from the adjacency mapping
        # Since u_idx refers to sorted_nodes index, we can trace it if we have it.
        # But wait! If we don't have sorted_nodes in the signature, we can extract from edges_df instead:
        pass
        
    # Let's extract row, col and costs directly from edges_df by mapping costs
    # Construct cost lookup from weighted_adj_dict
    # (u_idx, v_idx) -> cost
    # Wait, it's easier to iterate edges_df and get their endpoints:
    # First, let's build the cost map based on edges_df matched with weighted_adj_dict
    # We can reconstruct it or construct the heatmap based on the edge weights
    # To keep it extremely simple, let's represent the grid cell average cost:
    # We'll map the cost list to each cell coordinate
    for u_idx, neighbors in weighted_adj_dict.items():
        for v_idx, cost, dur, dist in neighbors:
            # We don't have sorted_nodes here, but we can compute or pass it.
            # Let's check how we can get cell from node index.
            # Actually, let's pass sorted_nodes to make it robust!
            pass
            
def plot_grid_edge_cost_heatmap(grid_dims, sorted_nodes, weighted_adj_dict, save_path):
    """
    Plot a spatial heatmap of average edge costs to visualize high-cost routing regions.
    """
    num_rows, num_cols = grid_dims
    heatmap = np.zeros((num_rows, num_cols))
    counts = np.zeros((num_rows, num_cols))
    
    for u_idx, neighbors in weighted_adj_dict.items():
        u_cell = sorted_nodes[u_idx]
        for v_idx, cost, dur, dist in neighbors:
            v_cell = sorted_nodes[v_idx]
            heatmap[v_cell[0], v_cell[1]] += cost
            counts[v_cell[0], v_cell[1]] += 1
            
            heatmap[u_cell[0], u_cell[1]] += cost
            counts[u_cell[0], u_cell[1]] += 1
            
    # Avoid division by zero
    valid_mask = counts > 0
    heatmap[valid_mask] = heatmap[valid_mask] / counts[valid_mask]
    # Set unvisited cells to a neutral/zero value
    heatmap[~valid_mask] = 0.0
    
    fig, ax = plt.subplots(figsize=(9, 8))
    # Plot using a beautiful sequential colormap (e.g. 'inferno' or 'magma')
    im = ax.imshow(heatmap, origin='lower', cmap='plasma', extent=[0, num_cols, 0, num_rows])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Average Transition Edge Cost", fontsize=11)
    
    ax.set_xticks(np.arange(0, num_cols + 1, 5))
    ax.set_yticks(np.arange(0, num_rows + 1, 5))
    ax.grid(True, which='both', color='white', linestyle='--', alpha=0.15)
    
    ax.set_xlabel("Grid Column")
    ax.set_ylabel("Grid Row")
    ax.set_title("Beijing Grid Traffic Edge-Cost Heatmap", fontsize=13, fontweight='bold')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
