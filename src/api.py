from fastapi import FastAPI
import tensorflow as tf
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

app = FastAPI()

# ✅ Load trained model
model = tf.keras.models.load_model("optimized_lstm_stock_model.h5")


# ✅ Predict function
@app.get("/predict/{stock_ticker}")
def predict(stock_ticker: str):
    df = pd.read_csv("final_dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by="Date")  # Ensure chronological order
    if df.empty:
        return {"error": "No stock data found"}

    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df.iloc[:, 1:])

    X_input = df_scaled[-60:].reshape(1, 60, df_scaled.shape[1])
    predicted_scaled = model.predict(X_input)
    predicted_price = scaler.inverse_transform([[0, 0, 0, predicted_scaled[0][0], 0]])[0][3]

    return {"predicted_price": round(predicted_price, 2)}
