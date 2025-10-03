import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Load data
df = pd.read_csv('stock_prices.csv')

# Preprocess data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1,1))

# Split data into training and testing datasets
training_size = int(len(scaled_data) * 0.8)
testing_size = len(scaled_data) - training_size
training_data = scaled_data[0:training_size,:]
testing_data = scaled_data[training_size:len(scaled_data),:]

# Prepare data for input into LSTM RNN model
def create_dataset(data, time_steps=1):
    X, Y = [], []
    for i in range(len(data)-time_steps-1):
        X.append(data[i:(i+time_steps), 0])
        Y.append(data[i+time_steps, 0])
    return np.array(X), np.array(Y)

time_steps = 60
X_train, Y_train = create_dataset(training_data, time_steps)
X_test, Y_test = create_dataset(testing_data, time_steps)

# Build LSTM RNN model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, Y_train, epochs=100, batch_size=32)

# Evaluate the model
train_loss = model.evaluate(X_train, Y_train)
test_loss = model.evaluate(X_test, Y_test)
print('Train Loss:', train_loss)
print('Test Loss:', test_loss)

# Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Plot results
plt.plot(df['Close'].values)
plt.plot(np.arange(training_size+time_steps, len(scaled_data)), predictions, color='red')
plt.show()

