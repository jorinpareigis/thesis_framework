import yfinance as yf
import pandas as pd
import os

os.makedirs("data", exist_ok=True)

ticker = "^GSPC"
file_path = "data/SP500_daily.csv"

print(f"Downloading {ticker} data...")
df = yf.download(ticker, period="max")

# Flatten the Multi-Index columns if they exist
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Isolate the target column. The 'Date' index is preserved automatically.
df = df[['Close']]

df.to_csv(file_path)
print(f"Data saved successfully to {file_path}")