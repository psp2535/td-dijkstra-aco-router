import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def analyze_and_visualize_embeddings(embeddings, sorted_nodes, congestion_csv_path, outputs_dir):
    """
    Perform checks on generated node embeddings:
    1. Check for NaN values.
    2. Compute L2 norm statistics.
    3. Project embeddings to 2D using PCA.
    4. Save a scatter plot colored by node average congestion level.
    5. Save statistics to embedding_statistics.json.
    """
    embeddings_np = embeddings.numpy()
    
    # 1. NaN Check
    has_nan = bool(np.isnan(embeddings_np).any())
    
    # 2. L2 Norm Statistics
    norms = np.linalg.norm(embeddings_np, ord=2, axis=1)
    norm_stats = {
        "mean": float(np.mean(norms)),
        "std": float(np.std(norms)),
        "min": float(np.min(norms)),
        "max": float(np.max(norms))
    }
    
    # 3. Read congestion stats for color-coding
    avg_congestion = {}
    if os.path.exists(congestion_csv_path):
        try:
            congestion_df = pd.read_csv(congestion_csv_path)
            if not congestion_df.empty:
                # Group by cell and calculate average congestion level
                cong_grp = congestion_df.groupby(["grid_row", "grid_col"])["congestion_level"].mean().to_dict()
                for k, v in cong_grp.items():
                    avg_congestion[k] = v
        except Exception as e:
            print(f"  [Warning] Failed to read congestion stats for coloring: {str(e)}")
            
    # Compile colors based on congestion (default to 0.0 if not measured)
    color_values = []
    for node in sorted_nodes:
        color_values.append(avg_congestion.get(node, 0.0))
    color_values = np.array(color_values)
    
    # 4. Project using PCA (fully deterministic with random_state)
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings_np)
    
    # Plotting
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        embeddings_2d[:, 0], 
        embeddings_2d[:, 1], 
        c=color_values, 
        cmap="viridis", 
        alpha=0.8, 
        edgecolors='none', 
        s=40
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label("Average Congestion Level (0=FreeFlow, 2=Congested)")
    
    plt.title("2D Projection of IM-VRM Node Embeddings (PCA)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(outputs_dir, "embeddings_visualization.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Compile final statistics
    stats = {
        "embedding_shape": list(embeddings_np.shape),
        "has_nans": has_nan,
        "l2_norm_stats": norm_stats,
        "pca_explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_]
    }
    
    # Save statistics as JSON
    stats_json_path = os.path.join(outputs_dir, "embedding_statistics.json")
    with open(stats_json_path, 'w') as f:
        json.dump(stats, f, indent=2)
        
    return stats, plot_path
