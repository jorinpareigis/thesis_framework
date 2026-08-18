import os
import urllib.request
import pandas as pd

def fetch_and_prepare_data() -> None:
    """
    Downloads the Beijing PM2.5 dataset from the UCI Machine Learning Repository,
    processes the temporal features into a datetime index, and handles missing values.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv"
    os.makedirs("data", exist_ok=True)
    temp_file = "data/temp_prsa.csv"
    final_file = "data/air_quality.csv"

    print("Downloading Beijing PM2.5 dataset from UCI...")
    urllib.request.urlretrieve(url, temp_file)

    print("Processing datetime and cleaning missing values...")
    df = pd.read_csv(temp_file)
    
    df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df = df.set_index('date')
    df = df[['pm2.5']]

    # Missing values must be imputed at the source to establish a clean ground truth 
    # before the framework's Monte Carlo corruption engine introduces synthetic anomalies.
    df['pm2.5'] = df['pm2.5'].ffill().bfill() 

    df.to_csv(final_file)
    os.remove(temp_file)
    
    print(f"Success! Cleaned dataset saved to {final_file} with {len(df)} hourly records.")

if __name__ == "__main__":
    fetch_and_prepare_data()