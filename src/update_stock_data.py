import pandas as pd
import yfinance as yf
from datetime import datetime
import json

#  Load existing dataset
df = pd.read_csv("stock_data.csv")

#  Get today's date
today = datetime.today().strftime("%Y-%m-%d")

#  Fetch latest stock data
# Load stock ticker from config file
with open("config.json", "r") as f:
    config = json.load(f)
stock_ticker = config["stock_ticker"]  # Read ticker

new_data = yf.download(stock_ticker, start="2025-03-03",  auto_adjust=True)

# Process new data
if not new_data.empty:
    new_data.columns = new_data.columns.droplevel(1)  # Drop the first level (Ticker row)
    new_data = new_data.reset_index()
    print(new_data.tail())
    # Append to dataset
    new_data['Date'] = pd.to_datetime(new_data['Date'])
    new_data = new_data.iloc[1:]

    # Now concatenate df2_data with df1
    print(df.tail())
    df['Date'] = pd.to_datetime(df['Date'])
    df = pd.concat([df, new_data], ignore_index=True)
    # df = pd.concat([df, new_data])
    print(df.tail())
    df.to_csv("stock_data.csv", index=False)
    print(" Stock data updated successfully!")
    '''print(new_data.head())
    new_data["SMA_50"] = new_data["Close"].rolling(window=50).mean()  # 50-day SMA
    new_data["SMA_200"] = new_data["Close"].rolling(window=200).mean()  # 200-day SMA

    #  MACD
    new_data["EMA_12"] = new_data["Close"].ewm(span=12, adjust=False).mean()
    new_data["EMA_26"] = new_data["Close"].ewm(span=26, adjust=False).mean()
    new_data["MACD"] = new_data["EMA_12"] - new_data["EMA_26"]
    new_data["Signal"] = new_data["MACD"].ewm(span=9, adjust=False).mean()

    #  RSI
    delta = new_data["Close"].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    new_data["RSI"] = 100 - (100 / (1 + rs))

    #  Bollinger Bands
    # Calculate 20-day Simple Moving Average (SMA)
    new_data["SMA_20"] = new_data["Close"].rolling(window=20).mean()

    # Calculate rolling standard deviation
    new_data["Rolling_STD"] = new_data["Close"].rolling(window=20).std()

    # Ensure there are no NaN values before assignment
    new_data["Rolling_STD"].fillna(0, inplace=True)

    # Compute Bollinger Bands
    new_data["BB_Upper"] = new_data["SMA_20"] + (new_data["Rolling_STD"] * 2)
    new_data["BB_Lower"] = new_data["SMA_20"] - (new_data["Rolling_STD"] * 2)

    # Drop intermediate column
    new_data.drop(columns=["Rolling_STD"], inplace=True)

    #  Stochastic Oscillator
    # Compute rolling lowest low (L14) and highest high (H14)
    new_data["L14"] = new_data["Low"].rolling(window=14, min_periods=1).min()
    new_data["H14"] = new_data["High"].rolling(window=14, min_periods=1).max()
    # Compute %K (Stochastic Oscillator)
    new_data["%K"] = 100 * ((new_data["Close"] - new_data["L14"]) / (new_data["H14"] - new_data["L14"] + 1e-9))  # Avoid division by zero

    # Compute %D (3-day moving average of %K)
    new_data["%D"] = new_data["%K"].rolling(window=3, min_periods=1).mean()

    #  Average True Range (ATR)
    new_data["High-Low"] = new_data["High"] - new_data["Low"]
    new_data["High-Close"] = abs(new_data["High"] - new_data["Close"].shift(1))
    new_data["Low-Close"] = abs(new_data["Low"] - new_data["Close"].shift(1))
    new_data["True Range"] = new_data[["High-Low", "High-Close", "Low-Close"]].max(axis=1)
    new_data["ATR"] = new_data["True Range"].rolling(window=14).mean()

    #  On-Balance Volume (OBV)
    new_data["OBV"] = (new_data["Volume"].where(new_data["Close"] > new_data["Close"].shift(1), -new_data["Volume"])).cumsum()

    #print(new_data.head())
    # Display final dataset
    #print(new_data[["Close", "Volume", "SMA_50", "SMA_200", "MACD", "Signal", "RSI", "BB_Upper", "BB_Lower", "%K", "%D", "ATR", "OBV"]].tail())
    '''

else:
    print(" No new data available for today.")
