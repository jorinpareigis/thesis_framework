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
        missing_pct (float): The percentage of data to corrupt (or intensity step).
        
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
            all_indices = np.random.permutation(corrupted_data.index)
            all_directions = np.random.choice([1, -1], size=len(corrupted_data))
            
            outlier_indices = all_indices[:num_outliers]
            directions = all_directions[:num_outliers]
            
            std_dev = train_data.std()
            shift = directions * corruption_cfg.intensity * std_dev
            
            corrupted_data.loc[outlier_indices] += shift

    elif corruption_type == "gaussian_noise":
        np.random.seed(cfg.seed)
        intensity_scalar = (missing_pct / cfg.corruption_end) * corruption_cfg.max_intensity
        
        data_std = train_data.std()
        noise_std = data_std * intensity_scalar
        
        noise_array = np.random.normal(loc=0.0, scale=noise_std, size=len(corrupted_data))
        corrupted_data += noise_array

    elif corruption_type == "sensor_outage":
        num_to_drop = int(len(corrupted_data) * (missing_pct / 100.0))
        if num_to_drop > 0:
            np.random.seed(cfg.seed)
            n_rows = len(corrupted_data)
            
            # Log-Normal parameters
            mu, sigma = 2.0, 1.0 
            
            # Dynamic cap: Minimum of 500 rows or 10% of the dataset
            max_block_size = min(500, max(1, int(n_rows * 0.10)))
            
            drop_indices = []
            seen_indices = set()
            
            while len(drop_indices) < num_to_drop:
                outage_length = int(np.ceil(np.random.lognormal(mu, sigma)))
                # Apply the dynamic cap
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
        # 1. Lock the seed to determine drift direction for this specific Monte Carlo run
        np.random.seed(cfg.seed)
        direction = np.random.choice([1, -1])
        
        # 2. Calculate progression (0.0 to 1.0)
        step_ratio = missing_pct / cfg.corruption_end
        
        n_rows = len(corrupted_data)
        
        # 3. Calculate Onset Point: 
        # Lower intensity starts late. 100% intensity starts at index 0.
        onset_idx = int((1.0 - step_ratio) * n_rows)
        
        # 4. Calculate Final Magnitude at the very end of the dataset
        data_std = train_data.std()
        max_std = corruption_cfg.get("max_drift_std", 3.0)
        final_magnitude = step_ratio * max_std * data_std * direction
        
        # 5. Apply linear degradation
        drift_length = n_rows - onset_idx
        if drift_length > 0:
            # np.linspace creates a smooth, linear slope from 0 to the target severity
            drift_array = np.linspace(0, final_magnitude, drift_length)
            
            # Inject the drift into the trailing portion of the dataset
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