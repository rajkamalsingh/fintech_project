import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import json
# Ask user for stock ticker
ticker = input("Enter the stock ticker (e.g., AAPL for Apple): ").upper()

# Save the ticker to a JSON file
with open("config.json", "w") as f:
    json.dump({"stock_ticker": ticker}, f)

# Define date range
start_date = "2020-01-01"
end_date = "2024-01-01"

# Fetch stock data
df = yf.download(ticker, start=start_date, end=end_date)
#df.to_csv("data.csv", index=False)
print(df.head())
df.columns = df.columns.droplevel(1)  # Drop the first level (Ticker row)
df= df.reset_index()
print(df.head())
# Check if data is available
if df.empty:
    print("⚠️ No data found for the given stock ticker. Please try again.")
else:
    # 📌 Moving Averages
    df["SMA_50"] = df["Close"].rolling(window=50).mean()  # 50-day SMA
    df["SMA_200"] = df["Close"].rolling(window=200).mean()  # 200-day SMA

    # 📌 MACD
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # 📌 RSI
    delta = df["Close"].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # 📌 Bollinger Bands
    # Calculate 20-day Simple Moving Average (SMA)
    df["SMA_20"] = df["Close"].rolling(window=20).mean()

    # Calculate rolling standard deviation
    df["Rolling_STD"] = df["Close"].rolling(window=20).std()

    # Ensure there are no NaN values before assignment
    df["Rolling_STD"].fillna(0, inplace=True)

    # Compute Bollinger Bands
    df["BB_Upper"] = df["SMA_20"] + (df["Rolling_STD"] * 2)
    df["BB_Lower"] = df["SMA_20"] - (df["Rolling_STD"] * 2)

    # Drop intermediate column (optional)
    df.drop(columns=["Rolling_STD"], inplace=True)

    # 📌 Stochastic Oscillator
    # Compute rolling lowest low (L14) and highest high (H14)

    print(df.head())
    df["L14"] = df["Low"].rolling(window=14, min_periods=1).min()
    df["H14"] = df["High"].rolling(window=14, min_periods=1).max()
    print(df.head())
    # Compute %K (Stochastic Oscillator)
    df["%K"] = 100 * ((df["Close"] - df["L14"]) / (df["H14"] - df["L14"] + 1e-9))  # Avoid division by zero

    # Compute %D (3-day moving average of %K)
    df["%D"] = df["%K"].rolling(window=3, min_periods=1).mean()

    # 📌 Average True Range (ATR)
    df["High-Low"] = df["High"] - df["Low"]
    df["High-Close"] = abs(df["High"] - df["Close"].shift(1))
    df["Low-Close"] = abs(df["Low"] - df["Close"].shift(1))
    df["True Range"] = df[["High-Low", "High-Close", "Low-Close"]].max(axis=1)
    df["ATR"] = df["True Range"].rolling(window=14).mean()

    # 📌 On-Balance Volume (OBV)
    df["OBV"] = (df["Volume"].where(df["Close"] > df["Close"].shift(1), -df["Volume"])).cumsum()

    #print(df.head())
    # Display final dataset
    print(df[["Close", "Volume", "SMA_50", "SMA_200", "MACD", "Signal", "RSI", "BB_Upper", "BB_Lower", "%K", "%D", "ATR", "OBV"]].tail())

    # 📊 Plot Closing Price + Bollinger Bands
    plt.figure(figsize=(12, 6))
    plt.plot(df["Close"], label="Closing Price", color="blue", alpha=0.5)
    plt.plot(df["BB_Upper"], label="Bollinger Upper", linestyle="dashed", color="red")
    plt.plot(df["BB_Lower"], label="Bollinger Lower", linestyle="dashed", color="green")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.title(f"{ticker} Bollinger Bands")
    plt.legend()
    plt.show()

    # 📊 Plot RSI
    plt.figure(figsize=(12, 4))
    plt.plot(df["RSI"], label="RSI", color="brown")
    plt.axhline(70, linestyle="dashed", color="red")  # Overbought threshold
    plt.axhline(30, linestyle="dashed", color="green")  # Oversold threshold
    plt.xlabel("Date")
    plt.ylabel("RSI Value")
    plt.title(f"{ticker} RSI Indicator")
    plt.legend()
    plt.show()

    # 📊 Plot MACD
    plt.figure(figsize=(12, 6))
    plt.plot(df["MACD"], label="MACD", color="purple")
    plt.plot(df["Signal"], label="Signal Line", color="orange")
    plt.xlabel("Date")
    plt.ylabel("MACD Value")
    plt.title(f"{ticker} MACD Indicator")
    plt.legend()
    plt.show()

    # 📊 Plot Stochastic Oscillator
    plt.figure(figsize=(12, 4))
    plt.plot(df["%K"], label="%K Line", color="blue")
    plt.plot(df["%D"], label="%D Signal Line", color="orange")
    plt.axhline(80, linestyle="dashed", color="red")  # Overbought
    plt.axhline(20, linestyle="dashed", color="green")  # Oversold
    plt.xlabel("Date")
    plt.ylabel("Stochastic Value")
    plt.title(f"{ticker} Stochastic Oscillator")
    plt.legend()
    plt.show()

# Save stock data to CSV
df.to_csv("stock_data.csv", index=False)

print("✅ Stock price data saved to stock_data.csv")