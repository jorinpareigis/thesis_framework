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
        missing_pct (float): The percentage of data to corrupt (0 to 100).
        
    Returns:
        pd.Series: The corrupted and imputed dataset.
    """

    # Early exit to prevent unnecessary computation during the baseline (0%) iteration.
    if missing_pct == 0:
        return train_data.copy()
        
    corruption_cfg = cfg.corruption
    corruption_type = corruption_cfg.type
    imputation_method = corruption_cfg.method
    
    # Isolate operations on a copy to prevent accidental mutation of the global training object.
    corrupted_data = train_data.copy()
    
    if corruption_type == "mcar":
        # MCAR (Missing Completely At Random) logic.
        num_to_drop = int(len(corrupted_data) * (missing_pct / 100.0))
        if num_to_drop > 0:
            # Re-initializing the seed guarantees deterministic index selection.
            # This enforces strict subsetting: the 100 indices dropped at 1% corruption 
            # are explicitly retained within the 200 indices dropped at 2% corruption.
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
            
            # 1. Generate full-length random arrays to guarantee static RNG progression
            # This prevents the RNG sequence from desynchronizing. An outlier generated 
            # as a positive spike (+1) at step N remains positive at step N+1.
            all_indices = np.random.permutation(corrupted_data.index)
            all_directions = np.random.choice([1, -1], size=len(corrupted_data))
            
            # 2. Subset the pre-generated arrays
            outlier_indices = all_indices[:num_outliers]
            directions = all_directions[:num_outliers]
            
            # Scale anomaly magnitude using standard deviation instead of raw integers.
            # This ensures the intensity parameter remains mathematically valid regardless 
            # of whether the underlying dataset measures temperature, stock prices, or megawatts.
            std_dev = train_data.std()
            shift = directions * corruption_cfg.intensity * std_dev
            
            corrupted_data.loc[outlier_indices] += shift

    else:
        raise NotImplementedError(f"Corruption type '{corruption_type}' is not recognized.")
    
    # Reconstruct the time-series continuity if required by the model.
    if imputation_method == "none":
        imputed_data = corrupted_data
    elif imputation_method == "linear_interpolation":
        # .bfill() and .ffill() act as boundary fallbacks. If the random choice drops 
        # the absolute first or last index in the series, interpolation fails (no endpoints 
        # to draw a line between). The fills patch these terminal edge cases.
        imputed_data = corrupted_data.interpolate(method='linear').bfill().ffill()
    elif imputation_method == "forward_fill":
        # .bfill() is chained as a fallback in case the very first index is corrupted.
        imputed_data = corrupted_data.ffill().bfill()
    else:
        raise NotImplementedError(f"Imputation method '{imputation_method}' is not recognized.")
        
    return imputed_data