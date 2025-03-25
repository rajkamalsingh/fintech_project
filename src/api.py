from fastapi import FastAPI
import tensorflow as tf
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

'''
# S3 bucket details
bucket_name = "your-model-bucket"
model_key = "optimized_lstm_stock_model.h5"
local_model_path = "/app/model/optimized_lstm_stock_model.h5"

# Ensure local directory exists
os.makedirs("/app/model", exist_ok=True)

# Download the model from S3
s3 = boto3.client('s3')
s3.download_file(bucket_name, model_key, local_model_path)

# Load the model
from keras.models import load_model
model = load_model(local_model_path)
'''
app = FastAPI()

# Load trained model
model = tf.keras.models.load_model("optimized_lstm_stock_model.h5")


# Predict function
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
