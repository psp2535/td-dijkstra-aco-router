import pandas as pd

def segment_trips(df, max_inactivity_sec=1200):
    """
    Segment a continuous taxi trajectory into multiple distinct trips.
    Whenever the time difference between consecutive records exceeds max_inactivity_sec,
    a new trip_id is created.
    
    Adds a 'trip_id' column to the DataFrame in the format '{taxi_id}_trip_{trip_number}'.
    """
    if df.empty:
        df = df.copy()
        df["trip_id"] = pd.Series(dtype=str)
        return df

    df = df.copy()
    
    # Calculate time differences between consecutive records in seconds
    time_diff_sec = df["timestamp"].diff().dt.total_seconds()
    
    # A new trip starts if the time difference is greater than the threshold or if it's the first record
    new_trip_mask = (time_diff_sec > max_inactivity_sec) | (time_diff_sec.isna())
    
    # Cumulative sum generates unique trip numbers for the taxi
    trip_numbers = new_trip_mask.cumsum()
    
    # Format trip_id: {taxi_id}_trip_{trip_number}
    taxi_id = df["taxi_id"].iloc[0]
    df["trip_id"] = [f"{taxi_id}_trip_{num}" for num in trip_numbers]
    
    return df
