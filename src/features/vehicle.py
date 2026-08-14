import numpy as np

def generate_synthetic_fleet(num_vehicles=50, depots=None, config=None):
    """
    Generate a synthetic fleet of delivery vehicles assigned to depots.
    Provides metadata for simulating multi-depot green routing.
    
    depots: list of dicts with keys 'depot_id', 'grid_row', 'grid_col'.
            If None, defaults to 4 depots placed near corners of the spatial grid.
    config: Loaded YAML configuration containing vehicle_params.
    """
    if depots is None:
        # Default depots placed at coordinates in grid space (assuming a 30x30 grid)
        depots = [
            {"depot_id": "depot_1", "grid_row": 5, "grid_col": 5},
            {"depot_id": "depot_2", "grid_row": 5, "grid_col": 24},
            {"depot_id": "depot_3", "grid_row": 24, "grid_col": 5},
            {"depot_id": "depot_4", "grid_row": 24, "grid_col": 24}
        ]
        
    # Get parameters from configuration, or fall back to defaults
    light_params = {
        "mass_kg": 1500,
        "base_fuel_rate_l_per_100km": 7.0,
        "load_penalty_factor": 0.05,
        "co2_g_per_liter": 2300.0
    }
    heavy_params = {
        "mass_kg": 8000,
        "base_fuel_rate_l_per_100km": 25.0,
        "load_penalty_factor": 0.08,
        "co2_g_per_liter": 2600.0
    }
    
    if config and "vehicle_params" in config:
        light_params.update(config["vehicle_params"].get("light_duty", {}))
        heavy_params.update(config["vehicle_params"].get("heavy_duty", {}))
        
    np.random.seed(config.get("seed", 42) if config else 42)
    
    fleet = []
    for i in range(num_vehicles):
        v_id = f"vehicle_{i+1:03d}"
        
        # Randomly select a depot
        depot = np.random.choice(depots)
        
        # Assign vehicle type (e.g. 70% light-duty, 30% heavy-duty)
        v_type = "light_duty" if np.random.rand() < 0.7 else "heavy_duty"
        
        # Max payload capacity limits
        max_payload = 1000.0 if v_type == "light_duty" else 8000.0
        
        # Pack metadata
        v_meta = {
            "vehicle_id": v_id,
            "type": v_type,
            "depot_id": depot["depot_id"],
            "depot_grid_row": depot["grid_row"],
            "depot_grid_col": depot["grid_col"],
            "max_payload_kg": max_payload,
            # Assign vehicle coefficients
            "config": light_params if v_type == "light_duty" else heavy_params
        }
        fleet.append(v_meta)
        
    return fleet, depots
