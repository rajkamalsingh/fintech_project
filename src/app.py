import requests, os
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

MODEL_PATH = os.path.join(os.getcwd(), "src/optimized_lstm_stock_model.h5")
DATASET_PATH = os.path.join(os.getcwd(), "src/final_dataset.csv")
print(MODEL_PATH)
print(os.path.exists(MODEL_PATH))
print(os.path.exists(DATASET_PATH))
print("Current Working Directory:", os.getcwd())
# ✅ Auto-reload the latest model
def load_latest_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    else:
        print("❌ Model file missing!")
        return None

model = load_latest_model()

# ✅ Load the model
#model = tf.keras.models.load_model(MODEL_PATH)

# ✅ Load stock data
df = pd.read_csv(DATASET_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(by="Date")

# ✅ Select features for prediction
features = ["Open", "High", "Low", "Close", "Volume"]
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[features])

# ✅ Function to predict tomorrow’s stock price
def predict_next_day():
    time_steps = 60
    X_input = df_scaled[-time_steps:]  # Last 60 days of data
    X_input = np.expand_dims(X_input, axis=0)  # Reshape for model input
    predicted_scaled = model.predict(X_input)
    predicted_price = scaler.inverse_transform([[0, 0, 0, predicted_scaled[0][0], 0]])[0][3]
    return predicted_price

# ✅ Streamlit UI
st.title("📈 Stock Price Prediction App")
st.write("Enter a stock ticker and get the predicted stock price for tomorrow.")

# ✅ Predict and display result
if st.button("Predict Tomorrow's Price"):
    predicted_price = predict_next_day()
    st.success(f"📊 Predicted Stock Price for Tomorrow: **${predicted_price:.2f}**")

    # ✅ Plot actual vs predicted prices
    st.subheader("📉 Actual vs Predicted Stock Prices")
    y_pred_scaled = model.predict(df_scaled[-60:].reshape(1, 60, len(features)))
    y_pred = scaler.inverse_transform([[0, 0, 0, y_pred_scaled[0][0], 0]])[0][3]

    plt.figure(figsize=(10, 5))
    plt.plot(df["Date"].tail(60), df["Close"].tail(60), label="Actual Price", color="blue")
    plt.axhline(y=y_pred, color="red", linestyle="--", label="Predicted Price")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend()
    st.pyplot(plt)
