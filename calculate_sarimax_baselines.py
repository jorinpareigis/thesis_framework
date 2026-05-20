import pandas as pd
import pmdarima as pm
import logging

# Configure terminal output
logging.basicConfig(level=logging.INFO, format='%(message)s')

datasets = [
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
        "seasonal": False, # Explicitly disabled to prevent solver crash
        "subset": 2016
    }
]

def process_datasets():
    for ds in datasets:
        logging.info(f"========== Running auto_arima for {ds['name']} ==========")
        
        try:
            # 1. Load Data (mirroring src/data_loader.py)
            df = pd.read_csv(ds['file'], index_col=0, parse_dates=True)
            df = df.sort_index()
            df = df[~df.index.duplicated(keep='first')]
            df = df.asfreq(ds['freq'])
            ts_data = df[ds['target']].ffill().bfill()
            
            # 2. Slice the subset (taking the FIRST contiguous block for tuning)
            # This creates the "Quarantine Zone"
            tuning_size = ds['subset']
            train_data = ts_data.iloc[:tuning_size]
            
            # 3. Run Hyndman-Khandakar Algorithm
            logging.info(f"Starting step-wise AIC search. This may take a few minutes...\n")
            
            model = pm.auto_arima(
                train_data,
                seasonal=ds['seasonal'],
                m=ds['m'] if ds['seasonal'] else 1,
                stepwise=True,
                trace=True, # Prints the AIC score of each tested order
                error_action='ignore',
                suppress_warnings=True,
                n_jobs=1 
            )
            
            # 4. Output YAML formatted blocks
            logging.info(f"\n--- COPY TO configs/dataset/{ds['name']}.yaml ---")
            
            # Format order (p,d,q)
            logging.info("sarimax_order:")
            for val in model.order:
                logging.info(f"  - {val}")
            
            # Format seasonal_order (P,D,Q,s)
            if ds['seasonal']:
                logging.info("sarimax_seasonal_order:")
                for val in model.seasonal_order:
                    logging.info(f"  - {val}")
            else:
                logging.info("sarimax_seasonal_order:")
                logging.info("  - 0\n  - 0\n  - 0\n  - 0")
            logging.info("----------------------------------------------------\n")
            
        except FileNotFoundError:
            logging.error(f"Error: {ds['file']} not found. Run download scripts first.\n")
        except Exception as e:
            logging.error(f"Failed to process {ds['name']}: {e}\n")

if __name__ == "__main__":
    process_datasets()