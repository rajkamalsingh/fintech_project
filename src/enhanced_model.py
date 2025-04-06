import pandas as pd
import numpy as np
import tensorflow as tf
from Scripts.pywin32_postinstall import verbose
from scipy.ndimage import label
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Attention, Input, LayerNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.python.keras.saving.saved_model_experimental import sequential
import sys

if not hasattr(sys.stdout, "encoding") or sys.stdout.encoding is None:
    sys.stdout.encoding = "utf-8"


# ------------------------------
# Load Data
# ------------------------------
data = pd.read_csv('final_dataset.csv')

# Select relevant features
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'News_Sentiment', 'SMA_50', 'SMA_200', 'RSI', '%K', 'BB_Upper', 'BB_Lower', 'ATR', 'MACD', 'OBV', 'Signal']
data = data[features]
print(data.columns.get_loc('Close'))
#print(data.head())
#print(data.std())
# Scale the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

x,y =[],[]
# Preparing sequence for LSTM
def create_sequence(data, sequence_length=60):

    for i in range (len(data)-sequence_length):
        x.append(data[i : i + sequence_length])
        y.append(data[i + sequence_length, 3]) #focusing on close price
    return np.array(x), np.array(y)

sequence_length =60
x,y = create_sequence(scaled_data,sequence_length)
print(np.isnan(x).sum(), np.isinf(x).sum())
print(np.isnan(y).sum(), np.isinf(y).sum())
print("Target Std Dev:", np.std(y))
print("y min:", np.min(y))
print("y max:", np.max(y))
# Split into train and test data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, shuffle=False)

#-----------------------------------
# Building LSTM model
inputs = Input(shape=(x_train.shape[1], x_train.shape[2]))

#model = Sequential()

# Bidirectional LSTM layer 1
m = Bidirectional(LSTM(128, return_sequences = True, activation = 'relu'))(inputs)
m=Dropout(0.3)(m)


# Bidirectional LSTM layer 2
m = Bidirectional(LSTM(64, return_sequences = True))(m)
m = Dropout(0.3)(m)

# Attention mechanism
#query = Dense(64)(x_train)
#value = Dense(64)(x_train)
attention_output = Attention()([m,m])

# Layer normalization
attention_output = LayerNormalization()(attention_output)

# LSTM layer with attention
m = LSTM(100, return_sequences = False)(attention_output)
m = Dropout(0.3)(m)

# output layer
output = Dense(1)(m)

model = tf.keras.Model(inputs,output)

#-----------------------------

# Optimizer with learning rate scheduler
initial_learning_rate = 0.0005
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=10000, decay_rate = 0.9, staircase = True
)


optimizer = Adam(learning_rate = lr_schedule)
model.compile(optimizer = optimizer, loss = 'mse', metrics = ['mae'])


# Train the model
history = model.fit(x_train, y_train, epochs=200, batch_size = 32, validation_data = (x_test,y_test), verbose=2)

# Evaluate the model
loss, mae = model.evaluate(x_test, y_test)
print(f'Test loss:{loss}, Test mae: {mae}')

# save the model
model.save('enhanced_lstm_stock-model.h5')


# model prediction and visualization
y_pred = model.predict(x_test)

# inverse scale predictions
#y_pred_inv = scaler.inverse_transform(np.concatenate((x_test[:,-1,-1],y_pred.reshape(-1,1)), axis =1))[:,-1]
#y_test_inv = scaler.inverse_transform(np.concatenate((x_test[:,-1,-1],y_test.reshape(-1,1)), axis =1))[:,-1]
# Step 1: Create a dummy array with same number of features (16)
dummy_input = np.zeros((len(y_pred), scaler.data_min_.shape[0]))  # shape (253, 16)

# Step 2: Insert the predicted values into the column corresponding to 'close' price (e.g., column index 5 if it's 6th)
# Replace 5 with the actual index of 'close' used during feature scaling
dummy_input[:, 3] = y_pred.reshape(-1)

# Step 3: Inverse transform and extract the predicted 'close' price
y_pred_inv = scaler.inverse_transform(dummy_input)[:, 3]

# Similarly for y_test
dummy_input[:, 3] = y_test.reshape(-1)
y_test_inv = scaler.inverse_transform(dummy_input)[:, 3]


# Plot predictions v/s actual price
plt.figure(figsize=(12,6))
plt.plot(y_test_inv, label='Actual Price', color = 'blue')
plt.plot(y_pred_inv, label='Predicted Price', color = 'red')
plt.title('Stock price prediction with enhanced model')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
