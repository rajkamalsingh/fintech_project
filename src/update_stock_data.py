import pandas as pd
import yfinance as yf
from datetime import datetime
import json

# ✅ Load existing dataset
df = pd.read_csv("final_stock_dataset.csv")

# ✅ Get today's date
today = datetime.today().strftime("%Y-%m-%d")

# ✅ Fetch latest stock data
# Load stock ticker from config file
with open("config.json", "r") as f:
    config = json.load(f)
stock_ticker = config["stock_ticker"]  # Read ticker

new_data = yf.download(stock_ticker, start=today, end=today)

# ✅ Process new data
if not new_data.empty:
    new_data.reset_index(inplace=True)
    new_data = new_data[["Date", "Open", "High", "Low", "Close", "Volume"]]

    # ✅ Append to dataset
    df = pd.concat([df, new_data])
    df.to_csv("stock_data.csv", index=False)
    print("✅ Stock data updated successfully!")
else:
    print("⚠️ No new data available for today.")
