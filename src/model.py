import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

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
