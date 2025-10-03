import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Fetch stock data from Yahoo Finance
def fetch_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date,multi_level_index=False)
    return stock_data['Close']  # We're using closing prices

# Data preprocessing
def prepare_data(data, seq_length):
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    
    # Create sequences
    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        print (i)
        X.append(scaled_data[i-seq_length:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    
    # Train-test split (80-20)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Reshape for LSTM [samples, timesteps, features]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    return X_train, X_test, y_train, y_test, scaler

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.linear(out[:, -1, :])
        return out

# Training function
def train_model(model, X_train, y_train, num_epochs=100, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        inputs = torch.from_numpy(X_train).float().to(device)
        labels = torch.from_numpy(y_train).float().to(device)
        
        #inputs = inputs.squeeze (2)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.6f}')

# Main execution
if __name__ == "__main__":
    # Parameters
    stock_symbol = "TXN"  # Microsoft stock as example
    start_date = "2020-01-01"
    end_date = "2025-02-20"  # Current date as per your setup
    seq_length = 60
    
    # Get and prepare data
    stock_data = fetch_stock_data(stock_symbol, start_date, end_date)
    X_train, X_test, y_train, y_test, scaler = prepare_data(stock_data, seq_length)
    
    # Initialize model
    model = LSTMModel().to(device)
    
    # Train the model
    train_model(model, X_train, y_train)
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.from_numpy(X_test).float().to(device)
        predictions = model(X_test_tensor).cpu().numpy()
        
        # Inverse transform predictions and actual values
        predictions = scaler.inverse_transform(predictions)
        y_test_transformed = scaler.inverse_transform([y_test])
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((predictions - y_test_transformed.T) ** 2))
        print(f'Root Mean Squared Error: {rmse:.2f}')
    
    # Predict next day's price
    with torch.no_grad():
        last_sequence = X_test[-1].reshape((1, seq_length, 1))
        last_sequence_tensor = torch.from_numpy(last_sequence).float().to(device)
        next_day_pred = model(last_sequence_tensor).cpu().numpy()
        next_day_price = scaler.inverse_transform(next_day_pred)[0][0]
        print(f"Predicted price for next day: ${next_day_price:.2f}")
    
    # Plotting results
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data.index[-len(y_test):], y_test_transformed.T, label='Actual Price')
    plt.plot(stock_data.index[-len(y_test):], predictions, label='Predicted Price')
    plt.title(f'{stock_symbol} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()