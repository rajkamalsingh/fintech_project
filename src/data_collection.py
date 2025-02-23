import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Ask user for stock ticker
ticker = input("Enter the stock ticker (e.g., AAPL for Apple): ").upper()

# Define date range
start_date = "2020-01-01"
end_date = "2024-01-01"

# Fetch stock data
df = yf.download(ticker, start=start_date, end=end_date)

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

    # Assuming df is your original DataFrame
    # Reset the MultiIndex to make Ticker and Date regular columns
    df_reset = df.reset_index()
    df_reset.columns = ["_".join(col).strip() for col in df_reset.columns]

    # Step 3: Extract the Ticker (e.g., 'AAPL') from the column names
    # Example: If columns are ['Date', 'Close_AAPL', 'High_AAPL', ...], extract 'AAPL'
    ticker = df_reset.columns[1].split("_")[-1]  # Assumes the first column after 'Date' has the ticker
    df_reset["Ticker"] = ticker  # Add 'Ticker' column

    # Step 4: Rename columns to remove the ticker suffix (e.g., 'Close_AAPL' → 'Close')
    df_reset.columns = [col.replace(f"_{ticker}", "") for col in df_reset.columns]
    df_reset.columns = [col.replace(f"_", "") for col in df_reset.columns]

    # Calculate L14 (rolling lowest low) and H14 (rolling highest high)
    df_reset["L14"] = df_reset["Low"].rolling(window=14, min_periods=1).min()
    df_reset["H14"] = df_reset["High"].rolling(window=14, min_periods=1).max()

    # Compute %K (Stochastic Oscillator)
    df_reset["%K"] = 100 * (
            (df_reset["Close"] - df_reset["L14"]) /
            (df_reset["H14"] - df_reset["L14"] + 1e-9)  # Avoid division by zero
    )

    # Compute %D (3-day moving average of %K)
    df_reset["%D"] = df_reset["%K"].rolling(window=3, min_periods=1).mean()
    # Inspect the result
    #print(df_reset[["Date", "Close", "L14", "H14", "%K", "%D"]].head())

    # (Optional) Set index back to MultiIndex
    print(df_reset.head())
    df_reset = df_reset.set_index(["Ticker", "Date"])
    # Compute %K (Stochastic Oscillator)
    #df["%K"] = 100 * ((df["Close"] - df["L14"]) / (df["H14"] - df["L14"] + 1e-9))  # Avoid division by zero
    #print(df_reset.head())
    # Compute %D (3-day moving average of %K)
    #df["%D"] = df["%K"].rolling(window=3, min_periods=1).mean()
    #df[["L14", "H14", "%K", "%D"]] = df_reset[["L14", "H14", "%K", "%D"]]
    # Step 1: Reconstruct the hierarchical columns in df_reset
    ticker = df_reset.index.get_level_values("Ticker")[0]  # Get the Ticker (e.g., 'AAPL')
    df_reset.columns = pd.MultiIndex.from_tuples([(ticker, col) for col in df_reset.columns])
    print(df_reset.head())
    print(df.head())
    # Step 2: Merge the new columns into df
    df = pd.concat([df, df_reset[[(ticker, "L14"), (ticker, "H14"), (ticker, "%K"), (ticker, "%D")]]], axis=1)
    #print(df.head())
    #df = df.merge(df_reset[["L14", "H14", "%K", "%D"]], left_index=True, right_index=True, how="left")
    #print(df_final.head())

    print(df.head())

    # 📌 Average True Range (ATR)
    df["High-Low"] = df["High"] - df["Low"]
    df["High-Close"] = abs(df["High"] - df["Close"].shift(1))
    df["Low-Close"] = abs(df["Low"] - df["Close"].shift(1))
    df["True Range"] = df[["High-Low", "High-Close", "Low-Close"]].max(axis=1)
    df["ATR"] = df["True Range"].rolling(window=14).mean()

    # 📌 On-Balance Volume (OBV)
    df["OBV"] = (df["Volume"].where(df["Close"] > df["Close"].shift(1), -df["Volume"])).cumsum()

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
