import numpy as np
import pandas as pd

def coord_to_grid_cell(lat, lon, bbox, grid_dims):
    """
    Map coordinate arrays (latitude, longitude) to grid cell indices (row, col).
    bbox: [min_lat, min_lon, max_lat, max_lon]
    grid_dims: (num_rows, num_cols)
    Returns: (row_indices, col_indices)
    """
    min_lat, min_lon, max_lat, max_lon = bbox
    num_rows, num_cols = grid_dims
    
    # Calculate relative position ratio
    lat_ratio = (lat - min_lat) / (max_lat - min_lat)
    lon_ratio = (lon - min_lon) / (max_lon - min_lon)
    
    # Clip and floor to get grid coordinate
    row = np.clip(np.floor(lat_ratio * num_rows), 0, num_rows - 1).astype(int)
    col = np.clip(np.floor(lon_ratio * num_cols), 0, num_cols - 1).astype(int)
    
    return row, col

def compute_grid_congestion(df, bbox, grid_dims, free_flow_percentile=90, min_measurements=2, config=None):
    """
    Groups trajectory points by grid cell and hour to evaluate congestion.
    
    congestion classification:
    - ratio = average_speed / free_flow_speed
    - Level 0 (FreeFlow): ratio >= threshold_free (default 0.7)
    - Level 1 (Moderate): threshold_moderate <= ratio < threshold_free
    - Level 2 (Congested): ratio < threshold_moderate (default 0.4)
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
        
    # Get thresholds from config if available
    threshold_free = 0.7
    threshold_mod = 0.4
    if config and "spatial_grid" in config:
        threshold_free = config["spatial_grid"].get("congestion_threshold_free", 0.7)
        threshold_mod = config["spatial_grid"].get("congestion_threshold_moderate", 0.4)
        min_measurements = config["spatial_grid"].get("min_measurements", min_measurements)

    # 1. Map all points to grid coordinates
    rows, cols = coord_to_grid_cell(df["latitude"].values, df["longitude"].values, bbox, grid_dims)
    
    df = df.copy()
    df["grid_row"] = rows
    df["grid_col"] = cols
    df["hour"] = df["timestamp"].dt.hour
    
    # 2. Filter records that represent motion
    moving_df = df[(df["delta_time_sec"] > 0) & (df["speed_kmh"] > 0)].copy()
    
    if moving_df.empty:
        return pd.DataFrame(), pd.DataFrame()
        
    # 3. Calculate cell-wide free-flow speeds (90th percentile speed across all hours)
    cell_free_flow = moving_df.groupby(["grid_row", "grid_col"])["speed_kmh"].quantile(free_flow_percentile / 100.0)
    cell_free_flow = cell_free_flow.rename("free_flow_speed").reset_index()
    
    # Floor free flow speed at 5.0 km/h to prevent divide-by-zero or extremely high ratio artifacts
    cell_free_flow.loc[cell_free_flow["free_flow_speed"] < 5.0, "free_flow_speed"] = 5.0
    
    # 4. Calculate hourly average speed and measurement counts per cell
    hourly_stats = moving_df.groupby(["grid_row", "grid_col", "hour"]).agg(
        avg_speed=("speed_kmh", "mean"),
        count=("speed_kmh", "count")
    ).reset_index()
    
    # 5. Merge average speed and free-flow benchmarks
    grid_stats = pd.merge(hourly_stats, cell_free_flow, on=["grid_row", "grid_col"], how="left")
    
    # 6. Apply congestion criteria
    grid_stats["speed_ratio"] = grid_stats["avg_speed"] / grid_stats["free_flow_speed"]
    
    # Default is FreeFlow (0)
    grid_stats["congestion_level"] = 0
    
    # Categorize levels
    moderate_mask = (grid_stats["speed_ratio"] < threshold_free) & (grid_stats["speed_ratio"] >= threshold_mod)
    congested_mask = (grid_stats["speed_ratio"] < threshold_mod)
    
    grid_stats.loc[moderate_mask, "congestion_level"] = 1
    grid_stats.loc[congested_mask, "congestion_level"] = 2
    
    # Filter stats with very sparse measurements to default to FreeFlow (to avoid noise)
    sparse_mask = grid_stats["count"] < min_measurements
    grid_stats.loc[sparse_mask, "congestion_level"] = 0
    grid_stats.loc[sparse_mask, "speed_ratio"] = 1.0
    
    return grid_stats, cell_free_flow
