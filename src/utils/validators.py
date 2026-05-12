import logging

logger = logging.getLogger(__name__)

# Define the absolute rules for what corruptions make sense for what data
VALID_COMBINATIONS = {
    "sp500": [
        "mcar", 
        "outliers"
    ],
    "energy": [
        "mcar", 
        "outliers", 
        "sensor_outage",  # Smart meters can drop connection
        "gaussian_noise" # Meter calibration errors
    ],
    "iot_temp": [
        "mcar", 
        "outliers", 
        "sensor_outage", 
        "sensor_drift",   # Physical thermal drift
        "gaussian_noise"
    ],
    "air_quality": [
        "mcar", 
        "outliers", 
        "sensor_outage", 
        "sensor_drift",   # Dust buildup on lens
        "gaussian_noise"
    ]
}

def validate_configuration(cfg):
    """
    Checks the Hydra configuration for illogical dataset and corruption pairings.
    Raises a ValueError immediately if an invalid pair is detected.
    """
    dataset_name = cfg.dataset.name
    corruption_name = cfg.corruption.type
    
    # Check if the dataset exists in our rulebook
    if dataset_name not in VALID_COMBINATIONS:
        logger.warning(f"Dataset '{dataset_name}' is not strictly mapped in validators.py. Bypassing check.")
        return

    # Check if the requested corruption is allowed for this dataset
    allowed_corruptions = VALID_COMBINATIONS[dataset_name]
    
    if corruption_name not in allowed_corruptions:
        error_msg = (
            f"\n--- CONFIGURATION ERROR ---\n"
            f"Invalid combination: Cannot apply '{corruption_name}' to the '{dataset_name}' dataset.\n"
            f"Allowed corruptions for {dataset_name}: {', '.join(allowed_corruptions)}\n"
            f"---------------------------"
        )
        # We use ValueError so Hydra catches it and crashes before allocating any memory or GPUs
        raise ValueError(error_msg)