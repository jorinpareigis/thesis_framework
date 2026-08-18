import logging
import numpy as np
import pandas as pd
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

def apply_corruption(train_data: pd.Series, cfg: DictConfig, corruption_level: float) -> pd.Series:
    """
    Injects synthetic anomalies into the training data and applies the configured imputation method.

    Args:
        train_data (pd.Series): The clean historical time-series data.
        cfg (DictConfig): The Hydra configuration object containing corruption parameters.
        corruption_level (float): The intensity of the corruption on a scale from 0.0 to 100.0.

    Returns:
        pd.Series: The corrupted and subsequently imputed dataset, clipped to physical domain boundaries.
        
    Raises:
        NotImplementedError: If the specified corruption type or imputation method is unsupported.
    """
    if corruption_level == 0.0:
        return train_data.copy()
        
    corruption_type = cfg.corruption.type
    imputation_method = cfg.corruption.method
    
    corrupted_data = train_data.copy()
    num_rows = len(corrupted_data)
    
    if corruption_type == "mcar":
        num_to_drop = int(num_rows * (corruption_level / 100.0) * 0.40)
        if num_to_drop > 0:
            np.random.seed(cfg.current_corr_seed)
            drop_indices = np.random.choice(
                corrupted_data.index, 
                size=num_to_drop, 
                replace=False
            )
            corrupted_data.loc[drop_indices] = np.nan
            
    elif corruption_type == "outliers":
        num_outliers = int(num_rows * (corruption_level / 100.0) * 0.40)
        if num_outliers > 0:
            np.random.seed(cfg.current_corr_seed)
            all_indices = np.random.permutation(corrupted_data.index)
            
            if cfg.dataset.name == "air_quality":
                all_directions = np.ones(num_rows)
            else:
                all_directions = np.random.choice([1, -1], size=num_rows)
            
            outlier_indices = all_indices[:num_outliers]
            directions = all_directions[:num_outliers]
            
            std_dev = train_data.std()
            shift = directions * 5.0 * std_dev
            
            corrupted_data.loc[outlier_indices] += shift

    elif corruption_type == "gaussian_noise":
        np.random.seed(cfg.current_corr_seed)
        intensity_scalar = (corruption_level / 100.0) * 0.5
        
        data_std = train_data.std()
        noise_std = data_std * intensity_scalar
        
        noise_array = np.random.normal(loc=0.0, scale=noise_std, size=num_rows)
        corrupted_data += noise_array

    elif corruption_type == "sensor_outage":
        num_to_drop = int(num_rows * (corruption_level / 100.0) * 0.40)
        if num_to_drop > 0:
            np.random.seed(cfg.current_corr_seed)
            
            mu, sigma = 2.0, 1.0 
            max_block_size = min(500, max(1, int(num_rows * 0.10)))
            
            drop_indices = []
            seen_indices = set()
            
            while len(drop_indices) < num_to_drop:
                outage_length = int(np.ceil(np.random.lognormal(mu, sigma)))
                outage_length = min(outage_length, max_block_size)
                
                start_pos = np.random.randint(0, num_rows)
                end_pos = min(start_pos + outage_length, num_rows)
                
                for pos in range(start_pos, end_pos):
                    if pos not in seen_indices:
                        seen_indices.add(pos)
                        drop_indices.append(pos)
                        
                        if len(drop_indices) == num_to_drop:
                            break
                            
            corrupted_data.iloc[drop_indices] = np.nan
            
    elif corruption_type == "sensor_drift":
        np.random.seed(cfg.current_corr_seed)
        
        direction = 1 
        step_ratio = corruption_level / 100.0
        
        onset_idx = int((1.0 - step_ratio) * num_rows)
        data_std = train_data.std()
        final_magnitude = step_ratio * 3.0 * data_std * direction
        
        drift_length = num_rows - onset_idx
        if drift_length > 0:
            drift_array = np.linspace(0, final_magnitude, drift_length)
            corrupted_data.iloc[onset_idx:] += drift_array

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
        
    return imputed_data.clip(lower=0.0)