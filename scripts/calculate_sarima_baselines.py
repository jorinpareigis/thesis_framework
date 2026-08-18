import logging
from typing import Any

import pandas as pd
import pmdarima as pm

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

DATASETS: list[dict[str, Any]] = [
    {
        "name": "sp500",
        "file": "data/SP500_daily.csv",
        "target": "Close",
        "freq": "B",
        "m": 5,
        "seasonal": True,
        "subset": 1008
    },
    {
        "name": "energy",
        "file": "data/PJME_hourly.csv",
        "target": "PJME_MW",
        "freq": "h",
        "m": 24,
        "seasonal": True,
        "subset": 1680
    },
    {
        "name": "air_quality",
        "file": "data/air_quality.csv",
        "target": "pm2.5",
        "freq": "h",
        "m": 24,
        "seasonal": True,
        "subset": 1680
    },
    {
        "name": "iot_temp",
        "file": "data/iot_temp.csv",
        "target": "temperature",
        "freq": "5min",
        "m": 1, 
        # Explicitly disabled to prevent mathematical solver crashes on high-frequency data
        "seasonal": False, 
        "subset": 2016
    }
]

def process_datasets() -> None:
    """
    Executes the Hyndman-Khandakar algorithm to find the optimal SARIMA hyperparameters 
    for each configured dataset via step-wise AIC minimization.
    
    Outputs the resulting parameters in a YAML-compatible format intended for 
    direct insertion into the Hydra configuration files.
    """
    for dataset_cfg in DATASETS:
        logger.info(f"========== Running auto_arima for {dataset_cfg['name']} ==========")
        
        try:
            df = pd.read_csv(dataset_cfg['file'], index_col=0, parse_dates=True)
            df = df.sort_index()
            df = df[~df.index.duplicated(keep='first')]
            df = df.asfreq(dataset_cfg['freq'])
            ts_data = df[dataset_cfg['target']].ffill().bfill()
            
            # Isolate the tuning chunk to create a strict "Quarantine Zone". 
            # This ensures the Monte Carlo cross-validation loops in the main pipeline 
            # never test on the specific data segment used to discover these baselines.
            tuning_size = dataset_cfg['subset']
            train_data = ts_data.iloc[:tuning_size]
            
            logger.info("Starting step-wise AIC search. This may take a few minutes...\n")
            
            model = pm.auto_arima(
                train_data,
                seasonal=dataset_cfg['seasonal'],
                m=dataset_cfg['m'] if dataset_cfg['seasonal'] else 1,
                stepwise=True,
                trace=True,
                error_action='ignore',
                suppress_warnings=True,
                n_jobs=1 
            )
            
            logger.info(f"\n--- COPY TO configs/dataset/{dataset_cfg['name']}.yaml ---")
            
            logger.info("sarima_order:")
            for val in model.order:
                logger.info(f"  - {val}")
            
            if dataset_cfg['seasonal']:
                logger.info("sarima_seasonal_order:")
                for val in model.seasonal_order:
                    logger.info(f"  - {val}")
            else:
                logger.info("sarima_seasonal_order:")
                logger.info("  - 0\n  - 0\n  - 0\n  - 0")
            logger.info("----------------------------------------------------\n")
            
        except FileNotFoundError:
            logger.error(f"Error: {dataset_cfg['file']} not found. Run download scripts first.\n")
        except Exception as e:
            logger.error(f"Failed to process {dataset_cfg['name']}: {e}\n")

if __name__ == "__main__":
    process_datasets()