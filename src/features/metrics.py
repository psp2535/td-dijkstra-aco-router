import numpy as np
import pandas as pd
from src.utils.helpers import haversine_distance

def calculate_trip_kinematics(trip_df):
    """
    Calculate step-by-step kinematics (distance delta, time delta, velocity, acceleration)
    for a single continuous trip to avoid edge discontinuities.
    """
    trip_df = trip_df.copy()
    n = len(trip_df)
    
    # Defaults
    trip_df["delta_dist_m"] = 0.0
    trip_df["delta_time_sec"] = 0.0
    trip_df["speed_mps"] = 0.0
    trip_df["speed_kmh"] = 0.0
    trip_df["acceleration_m_s2"] = 0.0
    
    if n < 2:
        return trip_df
        
    # Get consecutive coordinate arrays
    lat1 = trip_df["latitude"].values[:-1]
    lon1 = trip_df["longitude"].values[:-1]
    lat2 = trip_df["latitude"].values[1:]
    lon2 = trip_df["longitude"].values[1:]
    
    # 1. Distance delta (meters)
    dists = haversine_distance(lat1, lon1, lat2, lon2)
    trip_df.loc[trip_df.index[1:], "delta_dist_m"] = dists
    
    # 2. Time delta (seconds)
    times = trip_df["timestamp"].values
    # Compute time differences in seconds
    time_diffs = (times[1:] - times[:-1]).astype('timedelta64[s]').astype(float)
    trip_df.loc[trip_df.index[1:], "delta_time_sec"] = time_diffs
    
    # 3. Instantaneous speed (m/s and km/h)
    speeds_mps = np.zeros(n)
    valid_time = time_diffs > 0
    # Avoid division by zero
    speeds_mps[1:][valid_time] = dists[valid_time] / time_diffs[valid_time]
    
    trip_df["speed_mps"] = speeds_mps
    trip_df["speed_kmh"] = speeds_mps * 3.6
    
    # 4. Acceleration (m/s^2)
    accel = np.zeros(n)
    speed_diffs = speeds_mps[1:] - speeds_mps[:-1]
    accel[1:][valid_time] = speed_diffs[valid_time] / time_diffs[valid_time]
    trip_df["acceleration_m_s2"] = accel
    
    return trip_df

def calculate_kinematics(df):
    """
    Group DataFrame by trip_id and calculate kinematics for each trip.
    """
    if df.empty or "trip_id" not in df.columns:
        return df
    return df.groupby("trip_id", group_keys=False).apply(calculate_trip_kinematics)

def compute_fuel_and_emissions(speed_kmh, delta_dist_m, delta_time_sec, vehicle_config, payload_kg=0.0):
    """
    Estimate fuel consumption (liters) and CO2 emissions (grams) for a single trajectory step.
    
    Formula model:
    - Base fuel rate is adjusted by payload weight.
    - Idling (speed = 0) consumes fuel at a fixed rate per second.
    - Travel speed adjusts fuel efficiency using a U-shaped efficiency multiplier
      (lower efficiency at crawl or high aerodynamic drag speeds).
    """
    base_rate = vehicle_config["base_fuel_rate_l_per_100km"]
    penalty_factor = vehicle_config["load_penalty_factor"]
    co2_g_per_l = vehicle_config["co2_g_per_liter"]
    
    # Adjust fuel consumption rate per 100km based on payload (unit of penalty is per 100kg load)
    # Adjusted rate = base_rate * (1 + penalty * (payload_kg / 100))
    adjusted_rate_l_100km = base_rate * (1.0 + penalty_factor * (payload_kg / 100.0))
    
    # Determine vehicle class base idle rate (liters per hour)
    is_heavy = vehicle_config.get("mass_kg", 1500) > 3000
    idle_rate_l_h = 3.5 if is_heavy else 1.2
    idle_rate_l_s = idle_rate_l_h / 3600.0
    
    # Vectorized fuel computation
    # Initialize output array
    fuel_l = np.zeros_like(speed_kmh, dtype=float)
    
    # Case 1: Idling (speed is zero or very close to zero, but time elapsed)
    idle_mask = (speed_kmh < 0.5) & (delta_time_sec > 0)
    fuel_l[idle_mask] = idle_rate_l_s * delta_time_sec[idle_mask]
    
    # Case 2: Moving
    moving_mask = speed_kmh >= 0.5
    v = speed_kmh[moving_mask]
    d_km = delta_dist_m[moving_mask] / 1000.0
    
    # U-shaped efficiency factor based on speed (v)
    eff_factor = np.ones_like(v)
    eff_factor[v < 10.0] = 2.0        # Crawl congestion
    eff_factor[(v >= 10.0) & (v < 30.0)] = 1.4  # Heavy traffic
    eff_factor[(v >= 30.0) & (v < 60.0)] = 1.0  # Optimum range
    eff_factor[(v >= 60.0) & (v < 90.0)] = 1.25 # Sub-optimum highway
    eff_factor[v >= 90.0] = 1.6        # Drag dominated high speed
    
    # Fuel consumed = (L/100km) / 100 * dist_km * efficiency_factor
    fuel_l[moving_mask] = (adjusted_rate_l_100km / 100.0) * d_km * eff_factor
    
    # Calculate CO2 emissions
    co2_g = fuel_l * co2_g_per_l
    
    return fuel_l, co2_g
