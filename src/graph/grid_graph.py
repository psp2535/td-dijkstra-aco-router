import pandas as pd
import numpy as np
from src.features.spatial_grid import coord_to_grid_cell

def build_grid_graph(clean_trips_df, grid_dims, bbox, min_transitions=2, config=None):
    """
    Construct a directed transition graph in grid cell space based on taxi movements.
    
    clean_trips_df: DataFrame containing cleaned trajectory points with trip_id, 
                    latitude, longitude, speed_kmh, and delta_time_sec.
    grid_dims: tuple (num_rows, num_cols)
    bbox: Beijing bounding box [min_lat, min_lon, max_lat, max_lon]
    min_transitions: minimum number of observed transitions to create an edge.
    
    Returns:
        nodes: dict of cell coordinates -> node metadata (average speed, active status)
        edges: DataFrame containing grid edges and traffic metrics (count, speed, duration)
    """
    if clean_trips_df.empty:
        return {}, pd.DataFrame()
        
    if config and "graph" in config:
        min_transitions = config["graph"].get("min_transitions", min_transitions)
        
    df = clean_trips_df.copy()
    
    # 1. Map GPS points to grid row and column if not already binned
    if "grid_row" not in df.columns or "grid_col" not in df.columns:
        rows, cols = coord_to_grid_cell(df["latitude"].values, df["longitude"].values, bbox, grid_dims)
        df["grid_row"] = rows
        df["grid_col"] = cols

    # 2. Shift columns inside each trip to identify transitions
    df["next_row"] = df.groupby("trip_id")["grid_row"].shift(-1)
    df["next_col"] = df.groupby("trip_id")["grid_col"].shift(-1)
    df["next_speed_kmh"] = df.groupby("trip_id")["speed_kmh"].shift(-1)
    df["next_time_sec"] = df.groupby("trip_id")["delta_time_sec"].shift(-1)
    
    # 3. Discard terminal points of trips (which have no 'next' cell)
    transitions = df.dropna(subset=["next_row", "next_col"]).copy()
    transitions["next_row"] = transitions["next_row"].astype(int)
    transitions["next_col"] = transitions["next_col"].astype(int)
    
    # 4. Filter out self-loops (points where the taxi remained in the same grid cell)
    inter_cell_transitions = transitions[
        (transitions["grid_row"] != transitions["next_row"]) |
        (transitions["grid_col"] != transitions["next_col"])
    ].copy()
    
    if inter_cell_transitions.empty:
        return {}, pd.DataFrame()
        
    # 5. Group by transition boundaries to aggregate statistics
    edges = inter_cell_transitions.groupby(
        ["grid_row", "grid_col", "next_row", "next_col"]
    ).agg(
        transition_count=("trip_id", "count"),
        avg_speed_kmh=("next_speed_kmh", "mean"),
        avg_duration_sec=("next_time_sec", "mean")
    ).reset_index()
    
    # 6. Apply filter to reduce graph noise
    if min_transitions > 1:
        edges = edges[edges["transition_count"] >= min_transitions].reset_index(drop=True)
        
    # 7. Collect node metadata (which grid cells were actually visited)
    visited_cells = set(df[["grid_row", "grid_col"]].itertuples(index=False, name=None))
    
    nodes = {}
    for r, c in visited_cells:
        # Calculate cell baseline statistics (e.g. average overall speed inside cell)
        cell_points = df[(df["grid_row"] == r) & (df["grid_col"] == c)]
        avg_speed = cell_points["speed_kmh"].mean() if not cell_points.empty else 0.0
        nodes[(r, c)] = {
            "grid_row": r,
            "grid_col": c,
            "avg_speed_kmh": avg_speed,
            "points_count": len(cell_points)
        }
        
    return nodes, edges
