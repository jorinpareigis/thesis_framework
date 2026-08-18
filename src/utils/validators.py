import logging
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

VALID_COMBINATIONS: dict[str, list[str]] = {
    "sp500": [
        "mcar", 
        "outliers"
    ],
    "energy": [
        "mcar", 
        "outliers", 
        "sensor_outage",
        "gaussian_noise"
    ],
    "iot_temp": [
        "mcar", 
        "outliers", 
        "sensor_outage", 
        "sensor_drift",
        "gaussian_noise"
    ],
    "air_quality": [
        "mcar", 
        "outliers", 
        "sensor_outage", 
        "sensor_drift",
        "gaussian_noise"
    ]
}

def validate_configuration(cfg: DictConfig) -> None:
    """
    Validates the Hydra configuration to prevent scientifically illogical pairings 
    of datasets and corruption methods.

    Args:
        cfg (DictConfig): The parsed Hydra configuration object.

    Raises:
        ValueError: If the requested corruption type is explicitly forbidden for the dataset.
    """
    dataset_name = cfg.dataset.name
    corruption_name = cfg.corruption.type
    
    if dataset_name not in VALID_COMBINATIONS:
        logger.warning(f"Dataset '{dataset_name}' is not strictly mapped in validators.py. Bypassing check.")
        return

    allowed_corruptions = VALID_COMBINATIONS[dataset_name]
    
    if corruption_name not in allowed_corruptions:
        # Raising ValueError ensures Hydra intercepts the crash during initialization
        error_msg = (
            f"\n--- CONFIGURATION ERROR ---\n"
            f"Invalid combination: Cannot apply '{corruption_name}' to the '{dataset_name}' dataset.\n"
            f"Allowed corruptions for {dataset_name}: {', '.join(allowed_corruptions)}\n"
            f"---------------------------"
        )
        raise ValueError(error_msg)