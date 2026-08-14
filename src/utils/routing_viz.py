import matplotlib.pyplot as plt
import numpy as np

def plot_greedy_route(grid_dims, route_cells, depot_cell, delivery_cells, save_path):
    """
    Plot the planned greedy route on the binned spatial grid network.
    
    grid_dims: tuple (num_rows, num_cols)
    route_cells: list of tuples (row, col) representing the full sequence of cells traversed
    depot_cell: tuple (row, col) of the starting depot
    delivery_cells: list of tuples (row, col) of the target delivery locations
    save_path: filename to save the generated figure
    """
    num_rows, num_cols = grid_dims
    
    # Create empty grid map for background representation
    grid_map = np.zeros((num_rows, num_cols))
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw grid boundaries as background
    ax.imshow(grid_map, origin='lower', cmap='binary', extent=[0, num_cols, 0, num_rows], alpha=0.05)
    
    # Set grid line ticks
    ax.set_xticks(np.arange(0, num_cols + 1, 5))
    ax.set_yticks(np.arange(0, num_rows + 1, 5))
    ax.grid(True, which='both', color='gray', linestyle='--', alpha=0.3)
    
    # 1. Plot the binned grid path if route is generated
    if len(route_cells) > 1:
        route_array = np.array(route_cells)
        # Shift coordinate indexes by 0.5 to align markers with cell centroids
        ax.plot(
            route_array[:, 1] + 0.5, 
            route_array[:, 0] + 0.5, 
            color='darkorange', 
            linewidth=2.5, 
            label='Planned Route Path',
            zorder=2
        )
        # Draw path direction arrows for transitions
        for i in range(len(route_cells) - 1):
            start = route_cells[i]
            end = route_cells[i+1]
            dx = (end[1] - start[1])
            dy = (end[0] - start[0])
            if dx != 0 or dy != 0:
                ax.annotate(
                    "", 
                    xy=(end[1] + 0.5, end[0] + 0.5), 
                    xytext=(start[1] + 0.5, start[0] + 0.5),
                    arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.5),
                    zorder=3
                )
                
    # 2. Plot delivery targets
    deliv_array = np.array(delivery_cells)
    if len(deliv_array) > 0:
        ax.scatter(
            deliv_array[:, 1] + 0.5, 
            deliv_array[:, 0] + 0.5, 
            color='forestgreen', 
            s=120, 
            marker='o', 
            label='Delivery Targets', 
            zorder=4
        )
        
    # 3. Plot start depot
    ax.scatter(
        depot_cell[1] + 0.5, 
        depot_cell[0] + 0.5, 
        color='crimson', 
        s=200, 
        marker='*', 
        label='Start Depot', 
        zorder=5
    )
    
    ax.set_xlim(0, num_cols)
    ax.set_ylim(0, num_rows)
    ax.set_xlabel("Grid Column index")
    ax.set_ylabel("Grid Row index")
    ax.set_title("GNN-Guided Greedy Route (Dijkstra-Lite Transitions)")
    ax.legend(loc='upper right')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
