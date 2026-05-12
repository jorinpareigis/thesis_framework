import pandas as pd
import os
import urllib.request

def fetch_and_prepare_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv"
    os.makedirs("data", exist_ok=True)
    temp_file = "data/temp_prsa.csv"
    final_file = "data/air_quality.csv"

    print("Downloading Beijing PM2.5 dataset from UCI...")
    urllib.request.urlretrieve(url, temp_file)

    print("Processing datetime and cleaning missing values...")
    df = pd.read_csv(temp_file)
    
    # Combine individual time columns into a single pandas datetime object
    df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df = df.set_index('date')

    # Isolate the target variable
    df = df[['pm2.5']]

    # CRITICAL: The raw dataset contains natural NaNs. 
    # We MUST fill these here so the framework starts with a clean "ground truth" 
    # before your corruptions.py logic starts intentionally breaking it.
    df['pm2.5'] = df['pm2.5'].ffill().bfill() 

    # Save to the final format expected by your data_loader
    df.to_csv(final_file)
    os.remove(temp_file)
    
    print(f"Success! Cleaned dataset saved to {final_file} with {len(df)} hourly records.")

if __name__ == "__main__":
    fetch_and_prepare_data()