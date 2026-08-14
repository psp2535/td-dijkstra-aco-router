import matplotlib.pyplot as plt
import numpy as np

def plot_pheromone_heatmap(pheromone_matrix, labels, save_path):
    """
    Renders a heatmap showing the final pheromone intensity between depot and delivery nodes.
    """
    matrix = np.array(pheromone_matrix)
    K = len(matrix)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="YlOrRd", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Pheromone Strength $\\tau$", fontsize=11)
    
    ax.set_xticks(np.arange(K))
    ax.set_yticks(np.arange(K))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    
    # Loop over data dimensions and create text annotations for numerical values
    for i in range(K):
        for j in range(K):
            if i != j:
                ax.text(j, i, f"{matrix[i, j]:.2f}",
                        ha="center", va="center",
                        color="white" if matrix[i, j] > matrix.max() / 2. else "black",
                        fontsize=9)
                
    ax.set_title("Final ACO Target Transition Pheromone Matrix", fontsize=12, fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_aco_convergence(history, save_path):
    """
    Draws best and mean objective value convergence curves over ACO iterations.
    """
    iterations = [h["iteration"] for h in history]
    best_objs = [h["best_objective"] for h in history]
    mean_objs = [h["mean_objective"] for h in history]
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    # Plot objectives
    ax1.plot(iterations, best_objs, color="forestgreen", marker="o", linewidth=2, label="Best Objective")
    ax1.plot(iterations, mean_objs, color="forestgreen", linestyle="--", alpha=0.6, label="Mean Objective")
    ax1.set_xlabel("Iteration", fontsize=11)
    ax1.set_ylabel("Multi-Objective Cost (normalized)", color="forestgreen", fontsize=11)
    ax1.tick_params(axis='y', labelcolor='forestgreen')
    ax1.grid(True, linestyle="--", alpha=0.3)
    
    # Add secondary axis for physical parameters (distance/congestion) if present
    if len(history) > 0 and "best_distance_km" in history[0]:
        ax2 = ax1.twinx()
        best_dists = [h["best_distance_km"] for h in history]
        ax2.plot(iterations, best_dists, color="dodgerblue", marker="s", alpha=0.8, linewidth=1.5, label="Best Distance (km)")
        ax2.set_ylabel("Best Route Distance (km)", color="dodgerblue", fontsize=11)
        ax2.tick_params(axis='y', labelcolor='dodgerblue')
        
    lines1, labels1 = ax1.get_legend_handles_labels()
    if 'ax2' in locals():
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    else:
        ax1.legend(loc="upper right")
        
    plt.title("Ant Colony Optimization Convergence History", fontsize=12, fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_aco_route_comparison(grid_dims, greedy_route, aco_route, depot, deliveries, save_path):
    """
    Plots an overlapping comparative map of Greedy (before) vs ACO (after) route paths.
    """
    num_rows, num_cols = grid_dims
    grid_map = np.zeros((num_rows, num_cols))
    
    fig, ax = plt.subplots(figsize=(9, 9))
    
    ax.imshow(grid_map, origin='lower', cmap='binary', extent=[0, num_cols, 0, num_rows], alpha=0.03)
    ax.set_xticks(np.arange(0, num_cols + 1, 5))
    ax.set_yticks(np.arange(0, num_rows + 1, 5))
    ax.grid(True, which='both', color='gray', linestyle='--', alpha=0.2)
    
    # 1. Plot Greedy route (dashed gray)
    if len(greedy_route) > 1:
        greedy_arr = np.array(greedy_route)
        ax.plot(
            greedy_arr[:, 1] + 0.5,
            greedy_arr[:, 0] + 0.5,
            color='darkorange',
            linestyle='--',
            linewidth=2.0,
            alpha=0.7,
            label='Greedy Route Baseline',
            zorder=2
        )
        
    # 2. Plot ACO route (solid purple)
    if len(aco_route) > 1:
        aco_arr = np.array(aco_route)
        ax.plot(
            aco_arr[:, 1] + 0.5,
            aco_arr[:, 0] + 0.5,
            color='purple',
            linewidth=3.0,
            label='ACO Optimised Route',
            zorder=3
        )
        
        # Add transition arrows for ACO Route
        for i in range(len(aco_route) - 1):
            start = aco_route[i]
            end = aco_route[i+1]
            dx = end[1] - start[1]
            dy = end[0] - start[0]
            if dx != 0 or dy != 0:
                ax.annotate(
                    "",
                    xy=(end[1] + 0.5, end[0] + 0.5),
                    xytext=(start[1] + 0.5, start[0] + 0.5),
                    arrowprops=dict(arrowstyle="->", color="purple", lw=1.5, alpha=0.8),
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
    ax.set_title("Routing Comparison: GNN-Greedy vs GNN-ACO Selection", fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.95)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
