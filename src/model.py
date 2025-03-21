import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
from tensorflow.keras.optimizers import Adam

# Load stock price dataset
df = pd.read_csv("final_dataset.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(by="Date")  # Ensure chronological order

# Select multiple features for training
features = ["Open", "High", "Low", "Close", "Volume"]  # Add more if needed

# Normalize data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[features])

# Function to create LSTM sequences
def create_sequences(data, time_steps=60):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i : i + time_steps])
        y.append(data[i + time_steps, 3])  # Predicting "Close" price
    return np.array(X), np.array(y)

# Define time step (e.g., last 60 days)
time_steps = 60

# Prepare training data
X, y = create_sequences(df_scaled, time_steps)

# Split into training (80%) and testing (20%)
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# Reshape X for LSTM input (samples, time_steps, features)
X_train = X_train.reshape(X_train.shape[0], time_steps, len(features))
X_test = X_test.reshape(X_test.shape[0], time_steps, len(features))

# Build LSTM Model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(time_steps, len(features))),
    Dropout(0.3),  # Increased dropout to reduce overfitting
    LSTM(128, return_sequences=False),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dense(1)  # Predict single "Close" price
])

# Compile the model
optimizer = Adam(learning_rate=0.0005)  # Default was 0.001
model.compile(optimizer=optimizer, loss="mean_squared_error")

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Save the improved model
model.save("optimized_lstm_stock_model.h5")
print("Optimized model saved as optimized_lstm_stock_model.h5")

# Predict tomorrow's stock price
X_input = df_scaled[-time_steps:]  # Last 60 days of data
X_input = np.expand_dims(X_input, axis=0)  # Reshape for model input

predicted_scaled = model.predict(X_input)

# Convert back to actual price (use Close price index)
predicted_price = scaler.inverse_transform([[0, 0, 0, predicted_scaled[0][0], 0]])[0][3]

print(f"📈 Predicted Stock Price for Tomorrow: ${predicted_price:.2f}")

#  Plot training & validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.show()


# Make predictions on test set
# Make predictions on test data
y_pred_scaled = model.predict(X_test)

# Convert predictions back to original price scale
y_pred = scaler.inverse_transform([[0, 0, 0, pred[0], 0] for pred in y_pred_scaled])[:, 3]
y_actual = scaler.inverse_transform([[0, 0, 0, actual, 0] for actual in y_test])[:, 3]

# Plot actual vs predicted prices
plt.figure(figsize=(12, 6))
plt.plot(y_actual, label="Actual Price", color="blue")
plt.plot(y_pred, label="Predicted Price", color="red")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.title("Actual vs Predicted Stock Prices")
plt.show()

# Compute RMSE
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
print(f"📊 RMSE: {rmse:.2f}")

