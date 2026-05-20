import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def load_data(cfg):
    """
    Loads, cleans, and splits time-series data based on the Hydra configuration.
    
    Args:
        cfg: The Omegaconf dictionary provided by Hydra.
        
    Returns:
        tuple: A tuple containing:
            * train_data (pd.Series): Historical data for model training.
            * test_data (pd.Series): Ground truth data for model evaluation.
    """
    
    dataset_cfg = cfg.dataset
    file_path = dataset_cfg.file_path
    target_col = dataset_cfg.target_column
    test_size = dataset_cfg.test_size
    subset_size = dataset_cfg.get("subset_size", None)

    logger.info(f"Loading data from: {file_path}")
    
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    df = df.sort_index()

    # Remove duplicate timestamps (e.g., caused by Daylight Saving Time)
    df = df[~df.index.duplicated(keep='first')]

    data_freq = dataset_cfg.get("frequency", "h")
    df = df.asfreq(data_freq)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {file_path}.")
        
    # Extract the raw target column (Do NOT impute yet to prevent test leakage)
    ts_data = df[target_col]

    if subset_size is not None:
        total_len = len(ts_data)
        
        # Define the quarantine zone size based on the SARIMAX tuning chunk
        tuning_size = subset_size 
        
        if total_len > subset_size:
            # Lock the seed to ensure this run's data slice is strictly reproducible
            np.random.seed(cfg.seed)
            
            # Ensure the random start point leaves exactly enough room for the subset
            max_start_idx = total_len - subset_size
            
            # CRITICAL FIX 2: Ensure we don't pull from the SARIMAX tuning quarantine zone
            if max_start_idx <= tuning_size:
                raise ValueError(f"Dataset is too small to quarantine the first {tuning_size} rows and sample {subset_size} rows.")
            
            # Randomly sample a start index strictly AFTER the quarantine zone
            start_idx = np.random.randint(tuning_size, max_start_idx + 1)
            
            ts_data = ts_data.iloc[start_idx : start_idx + subset_size]
            logger.info(f"Data sliced from random index {start_idx} (safely past quarantine) using seed {cfg.seed}.")
        else:
            raise ValueError(f"Data length ({total_len}) <= subset_size ({subset_size}). Cannot perform quarantine and slice.")

    # CRITICAL FIX 1: Split the data BEFORE imputing to prevent future test values 
    # from leaking backwards into the training data via bfill()
    train_raw = ts_data.iloc[:-test_size]
    test_raw = ts_data.iloc[-test_size:]

    # Independently impute train and test data
    train_data = train_raw.ffill().bfill()
    test_data = test_raw.ffill().bfill()

    logger.info(f"Train set size: {len(train_data)} | Test set size: {len(test_data)}")

    return train_data, test_data