import os
import pandas as pd

def fetch_and_prepare_data() -> None:
    """
    Downloads the IoT Machine Temperature dataset from the Numenta Anomaly Benchmark (NAB),
    standardizes the temporal index, and enforces a strict 5-minute sampling frequency.
    """
    url = "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/machine_temperature_system_failure.csv"
    os.makedirs("data", exist_ok=True)
    final_file = "data/iot_temp.csv"

    print("Downloading IoT Machine Temperature dataset from Numenta NAB...")
    df = pd.read_csv(url)

    df = df.rename(columns={'timestamp': 'date', 'value': 'temperature'})
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    # Resolving duplicate timestamps is required before pandas can mathematically 
    # enforce the continuous 5-minute frequency grid without raising indexing errors.
    df = df[~df.index.duplicated(keep='first')]
    df = df.asfreq('5min') 
    
    df['temperature'] = df['temperature'].ffill().bfill()

    df.to_csv(final_file)
    print(f"Success! Cleaned dataset saved to {final_file} with {len(df)} 5-minute records.")

if __name__ == "__main__":
    fetch_and_prepare_data()