import os
import pandas as pd
import numpy as np
from src.utils.helpers import haversine_distance

def load_t_drive_file(filepath):
    """
    Load a single T-Drive taxi trajectory text file.
    Format: taxi_id, date time, longitude, latitude
    Example: 1,2008-02-02 15:36:08,116.51172,39.92123
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Trajectory file not found: {filepath}")
        
    try:
        # T-Drive files lack headers. Read as CSV and assign column names.
        df = pd.read_csv(
            filepath, 
            header=None, 
            names=["taxi_id", "timestamp", "longitude", "latitude"],
            parse_dates=["timestamp"]
        )
        
        # Sort chronologically
        df = df.sort_values(by="timestamp").reset_index(drop=True)
        return df
    except Exception as e:
        raise ValueError(f"Error loading {filepath}: {str(e)}")

def clean_trajectory(df, bbox, speed_limit_kmh=120.0):
    """
    Clean the trajectory dataset:
    1. Filter coordinates within spatial bounding box (bbox = [min_lat, min_lon, max_lat, max_lon]).
    2. Drop records with missing values.
    3. Drop records with duplicate timestamps (keeps first).
    4. Remove consecutive records implying speeds exceeding speed_limit_kmh.
    """
    if df.empty:
        return df
        
    # 1. Spatial bounding box filter
    min_lat, min_lon, max_lat, max_lon = bbox
    spatial_mask = (
        (df["latitude"] >= min_lat) & (df["latitude"] <= max_lat) &
        (df["longitude"] >= min_lon) & (df["longitude"] <= max_lon)
    )
    df = df[spatial_mask].copy()
    
    if df.empty:
        return df
        
    # 2. Drop NaN values
    df = df.dropna(subset=["timestamp", "longitude", "latitude"])
    
    # 3. Sort and drop duplicate timestamps
    df = df.sort_values(by="timestamp")
    df = df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    
    if len(df) < 2:
        return df

    # 4. Iteratively remove speed outliers. 
    # High frequency GPS logs can sometimes jump. If speed is physically impossible,
    # we discard the destination point of that transition.
    clean_indices = [0]
    
    for i in range(1, len(df)):
        prev_idx = clean_indices[-1]
        
        lat1, lon1 = df.loc[prev_idx, "latitude"], df.loc[prev_idx, "longitude"]
        lat2, lon2 = df.loc[i, "latitude"], df.loc[i, "longitude"]
        
        t1, t2 = df.loc[prev_idx, "timestamp"], df.loc[i, "timestamp"]
        time_diff_sec = (t2 - t1).total_seconds()
        
        if time_diff_sec <= 0:
            continue
            
        dist_m = haversine_distance(lat1, lon1, lat2, lon2)
        speed_mps = dist_m / time_diff_sec
        speed_kmh = speed_mps * 3.6
        
        if speed_kmh <= speed_limit_kmh:
            clean_indices.append(i)
            
    df_clean = df.iloc[clean_indices].reset_index(drop=True)
    return df_clean
