import os
import logging
import yaml
import numpy as np

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in meters between two points 
    on the earth (specified in decimal degrees).
    Supports scalars and numpy arrays.
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2.0 * np.arcsin(np.sqrt(a))
    
    # Radius of earth in meters
    r = 6371000.0
    return c * r

def load_config(config_path=None):
    """
    Load YAML configuration file. If config_path is not specified, 
    look for config/default_config.yaml.
    """
    if config_path is None:
        # Resolve path relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "../../config/default_config.yaml")
        config_path = os.path.normpath(config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_logging(log_dir="outputs", log_name="pipeline.log"):
    """
    Set up system loggers to print to console and write to file.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, log_name), mode='w', encoding='utf-8')
        ]
    )
    
    logger = logging.getLogger("IM-VRM")
    logger.info("Logging successfully initialized.")
    return logger
