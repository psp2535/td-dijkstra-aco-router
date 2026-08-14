import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def prepare_congestion_dataset(processed_dir):
    """
    Load data from grid stats, graph nodes, and graph edges to construct
    the features and target matrix for traffic congestion forecasting.
    
    Features built:
    - grid_row, grid_col (spatial coords)
    - hour (temporal)
    - traffic_count (traffic count in hour)
    - points_count (overall popularity)
    - baseline_speed (overall cell speed)
    - transition_density (in-degree + out-degree)
    - prev_congestion (lag-1 history value)
    
    Target:
    - congestion_level (0, 1, 2)
    """
    stats_csv = os.path.join(processed_dir, "grid_congestion_stats.csv")
    nodes_json = os.path.join(processed_dir, "graph_nodes.json")
    edges_csv = os.path.join(processed_dir, "graph_edges.csv")
    
    if not os.path.exists(stats_csv) or not os.path.exists(nodes_json) or not os.path.exists(edges_csv):
        raise FileNotFoundError("Required baseline preprocessing files are missing in data/processed/")
        
    grid_stats = pd.read_csv(stats_csv)
    edges_df = pd.read_csv(edges_csv)
    
    with open(nodes_json, 'r') as f:
        nodes_dict = json.load(f)
        
    # 1. Extract static node features from nodes dict
    node_features = []
    # Compute transition density from edges
    out_degrees = edges_df.groupby(["grid_row", "grid_col"])["transition_count"].sum().to_dict()
    in_degrees = edges_df.groupby(["next_row", "next_col"])["transition_count"].sum().to_dict()
    
    for k, v in nodes_dict.items():
        r, c = map(int, k.split(','))
        pts = float(v.get("points_count", 0.0))
        base_speed = float(v.get("avg_speed_kmh", 0.0))
        density = float(out_degrees.get((r, c), 0) + in_degrees.get((r, c), 0))
        
        node_features.append({
            "grid_row": r,
            "grid_col": c,
            "points_count": pts,
            "baseline_speed": base_speed,
            "transition_density": density
        })
    node_features_df = pd.DataFrame(node_features)
    
    # 2. Compute Congestion History (Lag-1 value): congestion_level at hour - 1 mod 24
    df_lag = grid_stats[["grid_row", "grid_col", "hour", "congestion_level"]].copy()
    # Map hour-1 to hour by shifting the key forward
    df_lag["hour"] = (df_lag["hour"] + 1) % 24
    df_lag = df_lag.rename(columns={"congestion_level": "prev_congestion"})
    
    # 3. Merge grid stats, node features, and lag-1 historical values
    dataset = pd.merge(grid_stats, node_features_df, on=["grid_row", "grid_col"], how="left")
    dataset = pd.merge(dataset, df_lag, on=["grid_row", "grid_col", "hour"], how="left")
    
    # Fill missing values for lag-1 history with 0 (FreeFlow)
    dataset["prev_congestion"] = dataset["prev_congestion"].fillna(0).astype(int)
    
    # Select clean feature matrix and target. 
    # NOTE: We exclude avg_speed, free_flow_speed, count, and speed_ratio to avoid direct data leaks.
    feature_cols = [
        "grid_row", "grid_col", "hour", "count", 
        "points_count", "baseline_speed", "transition_density", "prev_congestion"
    ]
    
    # Rename count to traffic_count for clarity
    dataset = dataset.rename(columns={"count": "traffic_count"})
    feature_cols[3] = "traffic_count"
    
    X = dataset[feature_cols].copy()
    y = dataset["congestion_level"].copy()
    
    return X, y

def train_tcfmu(X, y, config, outputs_dir):
    """
    Train XGBoost traffic congestion prediction classifier using paper parameters.
    Splits data temporally (training hours 0-18, test hours 19-23).
    """
    # Print class distribution (imbalance check)
    class_counts = y.value_counts().to_dict()
    total = len(y)
    print("Class Imbalance Statistics before training:")
    for lvl, count in class_counts.items():
        lbl = ["FreeFlow", "Moderate", "Congested"][lvl]
        print(f"  Level {lvl} ({lbl}): {count} entries ({count/total*100:.2f}%)")
        
    # Temporal Train/Test split: Train on hours 0-18 (~80%), test on 19-23 (~20%)
    train_mask = X["hour"] <= 18
    X_train = X[train_mask]
    y_train = y[train_mask]
    
    X_test = X[~train_mask]
    y_test = y[~train_mask]
    
    print(f"Dataset split: Train shape={X_train.shape}, Test shape={X_test.shape}")
    
    # Get hyperparameters
    seed = config["spatial_grid"].get("seed", 5) if config else 5
    n_est = 300  # Initial fast iteration count requested
    
    # Setup classifier using paper Table II specs
    model = xgb.XGBClassifier(
        n_estimators=n_est,
        learning_rate=0.5,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        random_state=seed,
        objective="multi:softprob",
        use_label_encoder=False,
        eval_metric="mlogloss"
    )
    
    model.fit(X_train, y_train)
    
    # Save model in json format
    model_json_path = os.path.join(outputs_dir, "xgboost_model.json")
    model.save_model(model_json_path)
    print(f"TCFMu model saved to: {model_json_path}")
    
    return model, X_train, y_train, X_test, y_test

def evaluate_and_plot_tcfmu(model, X_train, X_test, y_test, outputs_dir):
    """
    Compute accuracy, precision, recall, confusion matrix, and feature importances.
    Plot diagnostics.
    """
    y_pred = model.predict(X_test)
    
    # Calculations
    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    # Print per-class metrics
    print("\nPer-Class Classification Report:")
    for key, val in report.items():
        if key.isdigit():
            lbl = ["FreeFlow", "Moderate", "Congested"][int(key)]
            print(f"  Class {key} ({lbl}): Precision={val['precision']:.4f}, Recall={val['recall']:.4f}, F1-score={val['f1-score']:.4f}")
    print(f"  Overall Accuracy: {acc:.4f}")
    
    # Feature Importances
    importances = model.feature_importances_
    feature_names = X_train.columns.tolist()
    feat_imp = {feat: float(imp) for feat, imp in zip(feature_names, importances)}
    
    # Compile stats
    stats = {
        "accuracy": acc,
        "precision_macro": float(report["macro avg"]["precision"]),
        "recall_macro": float(report["macro avg"]["recall"]),
        "f1_macro": float(report["macro avg"]["f1-score"]),
        "class_metrics": {
            str(k): {
                "precision": float(v["precision"]),
                "recall": float(v["recall"]),
                "f1-score": float(v["f1-score"])
            } for k, v in report.items() if k.isdigit()
        },
        "confusion_matrix": cm.tolist(),
        "feature_importances": feat_imp
    }
    
    # Save statistics JSON
    metrics_json_path = os.path.join(outputs_dir, "xgboost_metrics.json")
    with open(metrics_json_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"TCFMu metrics saved to: {metrics_json_path}")
    
    # --- VISUALIZATIONS ---
    # 1. Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(np.unique(y_test))))
    ax.set_yticks(np.arange(len(np.unique(y_test))))
    ax.set_xticklabels(["FreeFlow", "Moderate", "Congested"])
    ax.set_yticklabels(["FreeFlow", "Moderate", "Congested"])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("TCFMu Congestion Confusion Matrix")
    
    # Loop over data dimensions and create text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2. else "black")
    plt.savefig(os.path.join(outputs_dir, "confusion_matrix.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Plot Feature Importance
    indices = np.argsort(importances)
    plt.figure(figsize=(8, 6))
    plt.barh(range(len(indices)), importances[indices], color="royalblue", align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.title("TCFMu Feature Importance")
    plt.grid(True, axis='x', linestyle='--', alpha=0.3)
    plt.savefig(os.path.join(outputs_dir, "feature_importance.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Plot Congestion Class Distribution
    classes = ["FreeFlow", "Moderate", "Congested"]
    counts = [y_test.value_counts().get(i, 0) for i in range(3)]
    plt.figure(figsize=(6, 4))
    plt.bar(classes, counts, color=["limegreen", "orange", "crimson"], edgecolor="black")
    plt.ylabel("Count")
    plt.title("TCFMu Test Class Distribution")
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    plt.savefig(os.path.join(outputs_dir, "congestion_class_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Diagnostic plots saved to {outputs_dir}/")
    return stats
