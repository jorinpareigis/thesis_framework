import os
import pandas as pd
import yfinance as yf

def fetch_and_prepare_data() -> None:
    """
    Downloads historical S&P 500 daily closing prices via the Yahoo Finance API,
    flattens potential MultiIndex structures, and exports the data.
    """
    os.makedirs("data", exist_ok=True)
    
    ticker = "^GSPC"
    file_path = "data/SP500_daily.csv"

    print(f"Downloading {ticker} data...")
    df = yf.download(ticker, period="max")

    # yfinance occasionally returns a MultiIndex DataFrame depending on the package version. 
    # Flattening guarantees structural consistency for the downstream data_loader.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Close']]

    df.to_csv(file_path)
    print(f"Data saved successfully to {file_path}")

if __name__ == "__main__":
    fetch_and_prepare_data()