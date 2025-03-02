import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, Dropout

# Load dataset
df = pd.read_csv("final_dataset.csv")
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)  # Set Date as index

# Select features for training (Close Price + Sentiment Scores)
features = ["Close", "Twitter_Sentiment", "News_Sentiment"]
df = df[features]

# Handle missing values
df.fillna(0, inplace=True)

# Normalize data using MinMaxScaler (scales values between 0 and 1)
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df)

# Convert back to DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=features, index=df.index)

# Display processed dataset
print(df_scaled.head())

# Function to create time-series sequences
def create_sequences(data, time_steps=60):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i : i + time_steps])  # Take past `time_steps` data
        y.append(data[i + time_steps, 0])   # Predict `Close` price at next step
    return np.array(X), np.array(y)

# Define sequence length (e.g., 60 days)
time_steps = 60

# Create sequences
X, y = create_sequences(df_scaled.values, time_steps)

# Split data into 80% training, 20% testing
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# Reshape X to fit LSTM model (samples, time_steps, features)
X_train = X_train.reshape(X_train.shape[0], time_steps, len(features))
X_test = X_test.reshape(X_test.shape[0], time_steps, len(features))

# Display dataset shape
print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")

# Build LSTM model
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(time_steps, len(features))),  # First LSTM layer
    Dropout(0.2),  # Prevent overfitting
    LSTM(100, return_sequences=False),  # Second LSTM layer
    Dropout(0.2),
    Dense(50, activation="relu"),  # Fully connected layer
    Dense(1)  # Output layer (Predict stock price)
])

# Compile the model
model.compile(optimizer="adam", loss="mean_squared_error")

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Save the trained model
model.save("lstm_stock_model.h5")

print(" Model training complete and saved as lstm_stock_model.h5")

