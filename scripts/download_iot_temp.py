import pandas as pd
import os

def fetch_and_prepare_data():
    url = "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/machine_temperature_system_failure.csv"
    os.makedirs("data", exist_ok=True)
    final_file = "data/iot_temp.csv"

    print("Downloading IoT Machine Temperature dataset from Numenta NAB...")
    df = pd.read_csv(url)

    # Standardize column names
    df = df.rename(columns={'timestamp': 'date', 'value': 'temperature'})
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    # FIX: Remove duplicate timestamps (keeping the first recorded value) 
    # before applying the strict frequency grid.
    df = df[~df.index.duplicated(keep='first')]

    # Now pandas can safely enforce the 5-minute intervals
    df = df.asfreq('5min') 
    df['temperature'] = df['temperature'].ffill().bfill()

    df.to_csv(final_file)
    print(f"Success! Cleaned dataset saved to {final_file} with {len(df)} 5-minute records.")

if __name__ == "__main__":
    fetch_and_prepare_data()