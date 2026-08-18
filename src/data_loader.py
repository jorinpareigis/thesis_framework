import logging
import numpy as np
import pandas as pd
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

def load_data(cfg: DictConfig) -> tuple[pd.Series, pd.Series]:
    """
    Loads, preprocesses, and splits time-series data into training and test sets.
    
    Enforces independent imputation to prevent data leakage and manages 
    deterministic subset sampling for Monte Carlo runs, strictly isolating 
    a pre-defined quarantine zone used for baseline model tuning.

    Args:
        cfg (DictConfig): The Hydra configuration object containing dataset parameters.

    Returns:
        tuple[pd.Series, pd.Series]: A tuple containing:
            - train_data (pd.Series): Historical data for model training.
            - test_data (pd.Series): Ground truth data for model evaluation.
            
    Raises:
        ValueError: If the target column is missing or the dataset is too small 
                    to support both the tuning quarantine zone and the required subset.
    """
    dataset_cfg = cfg.dataset
    file_path = dataset_cfg.file_path
    target_col = dataset_cfg.target_column
    test_size = dataset_cfg.test_size
    subset_size = dataset_cfg.get("subset_size", None)

    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    df = df.sort_index()

    # Daylight Saving Time transitions can create duplicate timestamps.
    df = df[~df.index.duplicated(keep='first')]
    df = df.asfreq(dataset_cfg.get("frequency", "h"))

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {file_path}.")
        
    ts_data = df[target_col]

    if subset_size is not None:
        total_len = len(ts_data)
        reserved_tuning_size = subset_size 
        
        if total_len > subset_size:
            np.random.seed(cfg.current_data_seed)
            
            max_start_idx = total_len - subset_size
            
            # Prevent Monte Carlo samples from overlapping with the deterministic 
            # SARIMA tuning data located at the start of the dataset.
            if max_start_idx <= reserved_tuning_size:
                raise ValueError(
                    f"Dataset too small to isolate {reserved_tuning_size} tuning rows "
                    f"and sample a sequence of {subset_size} rows."
                )
            
            start_idx = np.random.randint(reserved_tuning_size, max_start_idx + 1)
            ts_data = ts_data.iloc[start_idx : start_idx + subset_size]
            
            logger.info(f"Data sliced from random index {start_idx}.")
        else:
            raise ValueError(f"Data length ({total_len}) <= subset_size ({subset_size}).")

    # Imputation must occur AFTER splitting. Using bfill() on the entire dataset 
    # would allow future ground-truth values to leak into the training set.
    train_raw = ts_data.iloc[:-test_size]
    test_raw = ts_data.iloc[-test_size:]

    train_data = train_raw.ffill().bfill()
    test_data = test_raw.ffill().bfill()

    return train_data, test_data