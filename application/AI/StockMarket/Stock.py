import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import math
import yfinance as yf
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torch import optim

# --- 1. Device Configuration ---
# Check if CUDA is available and set the device accordingly.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# --- 2. Custom Scaling (Simplified for self-contained example) ---
class CustomMinMaxScaler:
    """Manual MinMax Scaler to fit and transform data, avoiding external dependencies."""
    def __init__(self):
        self.min = 0.0
        self.max = 1.0
        self.range = 1.0

    def fit(self, data):
        """Calculates min and max values from the training data."""
        data_flat = data.flatten()
        self.min = np.min(data_flat)
        self.max = np.max(data_flat)
        self.range = self.max - self.min
        # Handle cases where min == max to prevent division by zero
        if self.range == 0:
            self.range = 1.0

    def transform(self, data):
        """Applies the transformation."""
        return (data - self.min) / self.range

    def inverse_transform(self, data):
        """Reverts the transformation."""
        return data * self.range + self.min

# --- 3. Data Utilities ---

def download_stock_data(ticker_symbol='TXN', start_date='2020-01-01', end_date=None):
    """Downloads historical stock data using yfinance."""
    print(f"Downloading data for {ticker_symbol} from Yahoo Finance...")
    try:
        data = yf.download(ticker_symbol, start=start_date, end=end_date, auto_adjust = True)
        if data.empty:
            raise ValueError("No data returned from Yahoo Finance. Check ticker or date range.")
        
        # We will use the 'Close' price for univariate prediction
        close_prices = data['Close'].values.astype(np.float32).reshape(-1, 1)
        print(f"Successfully downloaded {len(close_prices)} data points.")
        return close_prices, data.index
    except Exception as e:
        print(f"Error downloading data: {e}")
        # Return empty data structures in case of failure
        return np.array([]).reshape(-1, 1), None

def create_sequences(data, seq_length):
    """
    Converts time series data into input sequences (X) and target values (y).
    X is a sequence of length `seq_length`, y is the next value.
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        # Input sequence: indices i to i + seq_length
        a = i
        b = i + seq_length
        seq = data[i:i + seq_length]
        X.append(data[i:i + seq_length])
        # Target value: index i + seq_length
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# --- 4. Model Definition ---

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1, dropout=0.2):
        """
        Initializes the LSTM model.
        
        Args:
            input_size (int): Number of features (1 for univariate stock price).
            hidden_size (int): Number of units in the hidden state of the LSTM.
            num_layers (int): Number of recurrent layers.
            output_size (int): Number of output predictions (1 for next day's price).
        """
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer: input shape (batch, seq_len, input_size) -> output (batch, seq_len, hidden_size)
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout
        )
        
        # Fully connected layer to map LSTM output to the desired prediction size
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        # Initialize hidden state (h0) and cell state (c0) on the correct device
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        
        # Forward propagate LSTM
        # out shape: (batch_size, sequence_length, hidden_size)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        
        # Use the output from the last time step for prediction
        out = self.fc(out[:, -1, :])
        
        return out

# --- 5. Visualization Function ---

def plot_results(dates, actual_prices, predicted_prices, train_size, seq_length):
    """Plots the actual vs. predicted prices."""
    
    # Calculate the starting index of the predicted data in the original time series
    # The first 'seq_length' days are used to create the first sequence, 
    # and training uses a subset of the data.
    plot_offset = train_size + seq_length

    plt.figure(figsize=(15, 7))
    plt.title('TXN Stock Price Prediction using PyTorch LSTM', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price (USD)', fontsize=14)

    # Plot full actual price history
    plt.plot(dates, actual_prices, label='Actual Price', color='#1f77b4', linewidth=2)
    
    # Create date index for predictions
    test_dates = dates[plot_offset:plot_offset + len(predicted_prices)]
    
    # Plot predicted test prices
    plt.plot(test_dates, predicted_prices, label='Predicted Price (Test Set)', color='#ff7f0e', linestyle='--', linewidth=2)

    # Highlight the split point between training and testing data
    train_end_date = dates[train_size + seq_length -1] # Adjust for sequence length offset
    plt.axvline(train_end_date, color='r', linestyle=':', label='Train/Test Split', alpha=0.6)

    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# --- 6. Main Execution Function ---

def main():
    # Hyperparameters and Settings
    TICKER_SYMBOL = 'TXN'
    SEQUENCE_LENGTH = 60  # Look back 60 days
    TRAIN_RATIO = 0.8     # 80% of sequenced data for training
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 300       # Increased epochs slightly for real data

    # 1. Download Data
    raw_data, dates = download_stock_data(TICKER_SYMBOL)
    if raw_data.size == 0:
        return

    # 2. Scale Data (Fit only on the portion that will be used for training)
    print("Scaling data...")
    scaler = CustomMinMaxScaler()
    
    # 3. Create Sequences
    X_seq_np, y_seq_np = create_sequences(raw_data, SEQUENCE_LENGTH)
    
    # The total number of sequences available after windowing
    total_sequences = len(X_seq_np)
    
    # Split point for sequences
    train_split_index = int(total_sequences * TRAIN_RATIO)
    
    # Slice the raw data and fit the scaler ONLY on the training portion
    # We must slice the raw data here to ensure the scaler is only fit on what the model "sees"
    # during training, which is the data up to the train_split_index + SEQUENCE_LENGTH
    scaler.fit(raw_data[:train_split_index + SEQUENCE_LENGTH])
    scaled_data = scaler.transform(raw_data)
    
    # Re-create sequences with the fully scaled data
    X_seq_np, y_seq_np = create_sequences(scaled_data, SEQUENCE_LENGTH)
    
    # Final split based on sequence arrays
    X_train_np, y_train_np = X_seq_np[:train_split_index], y_seq_np[:train_split_index]
    X_test_np, y_test_np = X_seq_np[train_split_index:], y_seq_np[train_split_index:]

    # 4. Convert to PyTorch Tensors and move to DEVICE
    X_train = torch.from_numpy(X_train_np).float().to(DEVICE)
    y_train = torch.from_numpy(y_train_np).float().to(DEVICE)
    X_test = torch.from_numpy(X_test_np).float().to(DEVICE)
    y_test = torch.from_numpy(y_test_np).float().to(DEVICE)
    
    # 5. Create DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 6. Initialize Model, Loss, and Optimizer
    input_dim = X_train.shape[2] # Should be 1
    model = LSTMModel(input_size=input_dim).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\nModel initialized on {DEVICE}. Starting training for {NUM_EPOCHS} epochs...")
    start_time = time.time()
    
    # 7. Training Loop
    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for i, (inputs, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Average Training Loss: {avg_loss:.6f}')

    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")

    # 8. Evaluation and Prediction
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test)
        test_loss = criterion(test_predictions, y_test)
        
        # Move tensors back to CPU for NumPy conversion and inverse scaling
        test_predictions_cpu = test_predictions.cpu().numpy()
        y_test_cpu = y_test.cpu().numpy()
        
        # Inverse transform the scaled predictions and actual values
        predicted_prices = scaler.inverse_transform(test_predictions_cpu)
        actual_test_prices = scaler.inverse_transform(y_test_cpu)
        
        # Calculate RMSE on the original scale
        rmse = np.sqrt(np.mean((predicted_prices - actual_test_prices) ** 2))

    print("\n--- Prediction Results ---")
    print(f"Test Loss (Scaled MSE): {test_loss.item():.6f}")
    print(f"Root Mean Squared Error (RMSE) on Original Scale: ${rmse:.2f}")
    
    print("\nExample Test Predictions (Last 5 Days):")
    for i in range(1, 6):
        idx = len(predicted_prices) - i
        print(f"  Day {idx+1} (Actual: ${actual_test_prices[idx][0]:.2f} | Predicted: ${predicted_prices[idx][0]:.2f})")
    
    # 9. Visualization
    print("\nDisplaying prediction results graph...")
    # Use the full raw data and the split index for plotting context
    plot_results(dates, raw_data.flatten(), predicted_prices.flatten(), train_split_index, SEQUENCE_LENGTH)

if __name__ == '__main__':
    # Ensure that all required libraries are installed (`pip install torch numpy pandas yfinance matplotlib`)
    main()