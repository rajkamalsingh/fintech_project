from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import uvicorn

app = FastAPI()

# Load scalers and model
scaler_features = joblib.load('scaler_features.pkl')
scaler_close = joblib.load('scaler_close.pkl')
model = tf.keras.models.load_model('enhanced_lstm_stock-model.h5')

# Define the feature list (same 15 used in enhanced_model.py)
FEATURES = ['Open', 'High', 'Low', 'Volume', 'News_Sentiment', 'SMA_50', 'SMA_200',
            'RSI', '%K', 'BB_Upper', 'BB_Lower', 'ATR', 'MACD', 'OBV', 'Signal']

@app.get("/predict")
def predict():
    # Load data
    df = pd.read_csv("final_dataset.csv")

    # Select relevant features and drop "Close"
    df = df[FEATURES]

    # Scale the data
    df_scaled = scaler_features.transform(df)

    # Get the last 120 steps
    last_120_steps = df_scaled[-120:]  # shape: (120, 15)

    if last_120_steps.shape != (120, 15):
        return {"error": f"Expected shape (120, 15), but got {last_120_steps.shape}"}

    x_input = np.expand_dims(last_120_steps, axis=0)  # shape: (1, 120, 15)

    # Make prediction
    predicted_scaled = model.predict(x_input)
    predicted_price = scaler_close.inverse_transform(predicted_scaled)[0][0]

    return {"predicted_close_price": round(predicted_price, 2)}

# Run with: uvicorn api_on_enhanced_model:app --reload
if __name__ == "__main__":
    uvicorn.run("api_on_enhanced_model:app", host="127.0.0.1", port=8000, reload=True)
