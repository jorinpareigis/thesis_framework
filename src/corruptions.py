import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def apply_corruption(train_data: pd.Series, cfg, missing_pct: float) -> pd.Series:
    """
    Injects anomalies into the training data and applies the configured imputation.
    
    Args:
        train_data (pd.Series): The clean historical data.
        cfg: The Hydra configuration dictionary.
        missing_pct (int): The percentage of data to corrupt (0 to 100).
        
    Returns:
        pd.Series: The corrupted and imputed dataset.
    """

    if missing_pct == 0:
        return train_data.copy()
        
    corruption_cfg = cfg.corruption
    corruption_type = corruption_cfg.type
    imputation_method = corruption_cfg.method
    
    corrupted_data = train_data.copy()
    
    if corruption_type == "mcar":
        num_to_drop = int(len(corrupted_data) * (missing_pct / 100.0))
        if num_to_drop > 0:
            np.random.seed(cfg.seed)
            drop_indices = np.random.choice(
                corrupted_data.index, 
                size=num_to_drop, 
                replace=False
            )
            corrupted_data.loc[drop_indices] = np.nan
    
    elif corruption_type == "outliers":
        num_outliers = int(len(corrupted_data) * (missing_pct / 100.0))
        if num_outliers > 0:
            np.random.seed(cfg.seed)
            outlier_indices = np.random.choice(
                corrupted_data.index, 
                size=num_outliers, 
                replace=False
            )
            
            std_dev = train_data.std()
            intensity = corruption_cfg.intensity
            
            directions = np.random.choice([1, -1], size=num_outliers)
            shift = directions * intensity * std_dev
            
            corrupted_data.loc[outlier_indices] += shift

    else:
        raise NotImplementedError(f"Corruption type '{corruption_type}' is not recognized.")
    
    if imputation_method == "none":
        imputed_data = corrupted_data
    elif imputation_method == "linear_interpolation":
        imputed_data = corrupted_data.interpolate(method='linear').bfill().ffill()
    elif imputation_method == "forward_fill":
        imputed_data = corrupted_data.ffill().bfill()
    else:
        raise NotImplementedError(f"Imputation method '{imputation_method}' is not recognized.")
        
    return imputed_data