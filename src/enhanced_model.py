import pandas as pd
import numpy as np
import tensorflow as tf
from Scripts.pywin32_postinstall import verbose
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Attention, Input, LayerNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.python.keras.saving.saved_model_experimental import sequential

# ------------------------------
# Load Data
# ------------------------------
data = pd.read_csv('final_dataset.csv')

# Select relevant features
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'News_Sentiment', 'SMA_50', 'SMA_200', 'RSI', '%K', 'BB_Upper', 'BB_Lower', 'ATR', 'MACD', 'OBV', 'Signal']
data = data[features]

# Scale the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)


# Preparing sequence for LSTM
def create_sequence(data, sequence_length=60):
    x,y =[],[]
    for i in range (len(data)-sequence_length):
        x.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length, 3]) #focusing on close price
    return np.array(x), np.array(y)

sequence_length =60
x,y = create_sequence(data,sequence_length)

# Split into train and test data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, shuffle=False)

#-----------------------------------
# Building LSTM model
model = Sequential()

# Bidirectional LSTM layer 1
model.add(Bidirectional(LSTM(128, return_sequences = True, activation = 'relu'),input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dropout(0.3))

# Bidirectional LSTM layer 2
model.add(Bidirectional(LSTM(64, return_sequences = True)))
model.add(Dropout(0.3))

# Attention mechanism
query = Dense(64)(x_train)
value = Dense(64)(x_train)
attention_output = Attention()([query,value])

# Layer normalization
attention_output = LayerNormalization()(attention_output)

# LSTM layer with attention
lstm_out = LSTM(100, return_sequences = False)(attention_output)
dropout = Dropout(0.3)(lstm_out)

# output layer
output = Dense(1)(dropout)
model = tf.keras.Model(inputs = model.input, outputs = output)

#-----------------------------

# Optimizer with learning rate scheduler
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=10000, decay_rate = 0.9, staircase = True
)


optimizer = Adam(learning_rate = lr_schedule)
model.compile(optimizer = optimizer, loss = 'mse', metrics = ['mae'])


# Train the model
history = model.fit(x_train, y_train, epochs=200, batch_size = 32, validation_data = (x_test,y_test), verbose=1)

# Evaluate the model
loss, mae = model.evaluate(x_test, y_test)
print(f'Test loss:{loss}, Test mae: {mae}')

# save the model
model.save('enhanced_lstm_stock-model.h5')


# model prediction and visualization
y_pred = model.predict(x_test)

# inverse scale predictions
y_pred_inv = scaler.inverse_transform(np.concatenate((x_test[:,-1,-1],y_pred.reshape(-1,1)), axis =1))[:,-1]
y_test_inv = scaler.inverse_transform(np.concatenate((x_test[:,-1,-1],y_test.reshape(-1,1)), axis =1))[:,-1]

# Plot predictions v/s actual price