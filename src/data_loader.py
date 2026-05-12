import pandas as pd
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
    # Extract dataset-specific parameters from the configuration
    dataset_cfg = cfg.dataset
    file_path = dataset_cfg.file_path
    target_col = dataset_cfg.target_column
    test_size = dataset_cfg.test_size

    # subset_size is optional. Using .get() prevents KeyErrors if omitted in the YAML.
    subset_size = dataset_cfg.get("subset_size", None)

    logger.info(f"Loading data from: {file_path}")
    
    # index_col=0 and parse_dates=True ensure pandas natively understands the time index,
    # which is required for plotting and statsmodels alignment.
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    # Force chronological sorting. If data is out of order, models could inadvertently 
    # train on future data (data leakage), invalidating the evaluation.
    df = df.sort_index()

    # CRITICAL ADDITION FOR PROPHET / SARIMAX:
    # Explicitly set the frequency to hourly. This fills in any completely missing 
    # timestamps with NaNs, which the ffill() step below will then patch.
    # Fetch the frequency from the YAML, defaulting to 'h' (hourly) for older datasets 
    # like energy, sp500, and air_quality if the key is not explicitly defined.
    data_freq = dataset_cfg.get("frequency", "h")
    df = df.asfreq(data_freq)

    # Fail-fast validation to catch configuration mapping errors immediately.
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {file_path}.")
        
    # Isolate the target variable as a 1D Series.
    # We use ffill() and bfill() instead of dropna(). Dropping rows physically removes 
    # timestamps, destroying the strict time-series frequency (e.g., turning a 24-hour 
    # cycle into a 23-hour cycle). This would break models relying on strict seasonal lags.
    ts_data = df[target_col].ffill().bfill()

    # If a subset is defined, slice from the end to evaluate against the most recent data.
    if subset_size is not None:
        ts_data = ts_data.iloc[-subset_size:]
        logger.info(f"Data truncated to the last {subset_size} rows.")

    # Sequential train/test split. Time-series data cannot be randomly split using 
    # scikit-learn's train_test_split. We must strictly isolate the final sequence.
    train_data = ts_data.iloc[:-test_size]
    test_data = ts_data.iloc[-test_size:]

    logger.info(f"Train set size: {len(train_data)} | Test set size: {len(test_data)}")

    return train_data, test_data