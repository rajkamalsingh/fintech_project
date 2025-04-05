import pandas as pd
import yfinance as yf
from datetime import datetime
import json

#  Load existing dataset
df = pd.read_csv("stock_data.csv")

#  Get today's date
today = datetime.today().strftime("%Y-%m-%d")
start_date = "2025-03-29"
#  Fetch latest stock data
# Load stock ticker from config file
with open("config.json", "r") as f:
    config = json.load(f)
stock_ticker = config["stock_ticker"]  # Read ticker

new_data = yf.download(stock_ticker, start=start_date,  auto_adjust=True)

# Process new data
if not new_data.empty:
    new_data.columns = new_data.columns.droplevel(1)  # Drop the first level (Ticker row)
    new_data = new_data.reset_index()
    print(new_data.tail())
    # Append to dataset
    new_data['Date'] = pd.to_datetime(new_data['Date'])
    #new_data = new_data.iloc[1:]
    print(new_data.tail())
    # Now concatenate df2_data with df1
    df['Date'] = pd.to_datetime(df['Date'])
    df = pd.concat([df, new_data], ignore_index=True)
    # df = pd.concat([df, new_data])
    print(df.tail())


    #print(new_data.head())
    df["SMA_50"] = df["Close"].rolling(window=50).mean()  # 50-day SMA
    df["SMA_200"] = df["Close"].rolling(window=200).mean()  # 200-day SMA

    #  MACD
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    #  RSI
    delta = df["Close"].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    #  Bollinger Bands
    # Calculate 20-day Simple Moving Average (SMA)
    df["SMA_20"] = df["Close"].rolling(window=20).mean()

    # Calculate rolling standard deviation
    df["Rolling_STD"] = df["Close"].rolling(window=20).std()

    # Ensure there are no NaN values before assignment
    df["Rolling_STD"].fillna(0, inplace=True)

    # Compute Bollinger Bands
    df["BB_Upper"] = df["SMA_20"] + (df["Rolling_STD"] * 2)
    df["BB_Lower"] = df["SMA_20"] - (df["Rolling_STD"] * 2)

    # Drop intermediate column
    df.drop(columns=["Rolling_STD"], inplace=True)

    #  Stochastic Oscillator
    # Compute rolling lowest low (L14) and highest high (H14)
    df["L14"] = df["Low"].rolling(window=14, min_periods=1).min()
    df["H14"] = df["High"].rolling(window=14, min_periods=1).max()
    # Compute %K (Stochastic Oscillator)
    df["%K"] = 100 * ((df["Close"] - df["L14"]) / (df["H14"] - df["L14"] + 1e-9))  # Avoid division by zero

    # Compute %D (3-day moving average of %K)
    df["%D"] = df["%K"].rolling(window=3, min_periods=1).mean()

    #  Average True Range (ATR)
    df["High-Low"] = df["High"] - df["Low"]
    df["High-Close"] = abs(df["High"] - df["Close"].shift(1))
    df["Low-Close"] = abs(df["Low"] - df["Close"].shift(1))
    df["True Range"] = df[["High-Low", "High-Close", "Low-Close"]].max(axis=1)
    df["ATR"] = df["True Range"].rolling(window=14).mean()

    #  On-Balance Volume (OBV)
    df["OBV"] = (df["Volume"].where(df["Close"] > df["Close"].shift(1), -df["Volume"])).cumsum()
    print(df.tail())
    #print(new_data.head())
    # Display final dataset
    #print(new_data[["Close", "Volume", "SMA_50", "SMA_200", "MACD", "Signal", "RSI", "BB_Upper", "BB_Lower", "%K", "%D", "ATR", "OBV"]].tail())
    df = df.fillna(0)
    df.to_csv("stock_data.csv", index=False)
    print(" Stock data updated successfully!")

else:
    print(" No new data available for today.")
