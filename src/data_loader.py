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
    dataset_cfg = cfg.dataset
    file_path = dataset_cfg.file_path
    target_col = dataset_cfg.target_column
    test_size = dataset_cfg.test_size

    subset_size = dataset_cfg.get("subset_size", None)

    logger.info(f"Loading data from: {file_path}")
    
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    df = df.sort_index()

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {file_path}.")
        
    ts_data = df[target_col].dropna()

    if subset_size is not None:
        ts_data = ts_data.iloc[-subset_size:]
        logger.info(f"Data truncated to the last {subset_size} rows.")

    train_data = ts_data.iloc[:-test_size]
    test_data = ts_data.iloc[-test_size:]

    logger.info(f"Train set size: {len(train_data)} | Test set size: {len(test_data)}")

    return train_data, test_data