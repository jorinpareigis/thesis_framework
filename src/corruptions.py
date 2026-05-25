import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def apply_corruption(train_data: pd.Series, cfg, corruption_level: float) -> pd.Series:
    """
    Injects anomalies into the training data and applies the configured imputation.
    
    Args:
        train_data (pd.Series): The clean historical data.
        cfg: The Hydra configuration dictionary.
        corruption_level (float): The intensity of the corruption on a 0-100 scale.
        
    Returns:
        pd.Series: The corrupted and imputed dataset.
    """
    if corruption_level == 0.0:
        return train_data.copy()
        
    corruption_cfg = cfg.corruption
    corruption_type = corruption_cfg.type
    imputation_method = corruption_cfg.method
    
    corrupted_data = train_data.copy()
    
    if corruption_type == "mcar":
        # 100% corruption_level equates to dropping 40% of the dataset
        num_to_drop = int(len(corrupted_data) * (corruption_level / 100.0) * 0.40)
        if num_to_drop > 0:
            np.random.seed(cfg.seed)
            drop_indices = np.random.choice(
                corrupted_data.index, 
                size=num_to_drop, 
                replace=False
            )
            corrupted_data.loc[drop_indices] = np.nan
    
    elif corruption_type == "outliers":
        # 100% corruption_level equates to 40% of rows becoming outliers
        num_outliers = int(len(corrupted_data) * (corruption_level / 100.0) * 0.40)
        if num_outliers > 0:
            np.random.seed(cfg.seed)
            all_indices = np.random.permutation(corrupted_data.index)
            all_directions = np.random.choice([1, -1], size=len(corrupted_data))
            
            outlier_indices = all_indices[:num_outliers]
            directions = all_directions[:num_outliers]
            
            std_dev = train_data.std()
            # Hardcoded multiplier of 5.0
            shift = directions * 5.0 * std_dev
            
            corrupted_data.loc[outlier_indices] += shift

    elif corruption_type == "gaussian_noise":
        np.random.seed(cfg.seed)
        # At 100% corruption, noise std_dev is 50% (0.5) of the dataset's std_dev
        intensity_scalar = (corruption_level / 100.0) * 0.5
        
        data_std = train_data.std()
        noise_std = data_std * intensity_scalar
        
        noise_array = np.random.normal(loc=0.0, scale=noise_std, size=len(corrupted_data))
        corrupted_data += noise_array

    elif corruption_type == "sensor_outage":
        # 100% corruption equates to 40% of the dataset dropped in blocks
        num_to_drop = int(len(corrupted_data) * (corruption_level / 100.0) * 0.40)
        if num_to_drop > 0:
            np.random.seed(cfg.seed)
            n_rows = len(corrupted_data)
            
            mu, sigma = 2.0, 1.0 
            max_block_size = min(500, max(1, int(n_rows * 0.10)))
            
            drop_indices = []
            seen_indices = set()
            
            while len(drop_indices) < num_to_drop:
                outage_length = int(np.ceil(np.random.lognormal(mu, sigma)))
                outage_length = min(outage_length, max_block_size)
                
                start_pos = np.random.randint(0, n_rows)
                end_pos = min(start_pos + outage_length, n_rows)
                
                for pos in range(start_pos, end_pos):
                    if pos not in seen_indices:
                        seen_indices.add(pos)
                        drop_indices.append(pos)
                        
                        if len(drop_indices) == num_to_drop:
                            break
                            
            corrupted_data.iloc[drop_indices] = np.nan
    
    elif corruption_type == "sensor_drift":
        np.random.seed(cfg.seed)
        direction = np.random.choice([1, -1])
        
        # Progression from 0.0 to 1.0
        step_ratio = corruption_level / 100.0
        n_rows = len(corrupted_data)
        
        onset_idx = int((1.0 - step_ratio) * n_rows)
        
        data_std = train_data.std()
        # Hardcoded max drift of 3.0 standard deviations
        final_magnitude = step_ratio * 3.0 * data_std * direction
        
        drift_length = n_rows - onset_idx
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
        
    return imputed_data