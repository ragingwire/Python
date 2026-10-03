import math
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# 1. Data Preprocessing & Feature Extraction
# ==========================================
class StockDataProcessor:
    """
    Handles downloading data via yfinance, calculating technical indicators,
    and formatting data into sequences for the PyTorch model.
    """
    def __init__(self, ticker, start_date, end_date, sequence_length=60):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.sequence_length = sequence_length
        self.scaler_features = MinMaxScaler(feature_range=(0, 1))
        self.scaler_target = MinMaxScaler(feature_range=(0, 1))
        self.raw_data = None
        self.processed_data = None

    def fetch_data(self):
        """Downloads historical data from yfinance."""
        print(f"Downloading data for {self.ticker}...")
        self.raw_data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        if self.raw_data.empty:
            raise ValueError(f"No data found for {self.ticker}. Check ticker symbol and dates.")
        
        # Flatten MultiIndex columns if yfinance returns them
        if isinstance(self.raw_data.columns, pd.MultiIndex):
             self.raw_data.columns = self.raw_data.columns.get_level_values(0)
             
        self.raw_data.dropna(inplace=True)
        print(f"Downloaded {len(self.raw_data)} trading days.")

    def add_technical_indicators(self):
        """Calculates MA10, EMA12, RSI, and MACD."""
        df = self.raw_data.copy()
        
        # 10-day Simple Moving Average
        df['MA10'] = df['Close'].rolling(window=10).mean()
        
        # 12-day Exponential Moving Average
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        
        # RSI (14-day)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD (EMA12 - EMA26)
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - ema26
        
        # Drop rows with NaN values resulting from rolling windows
        df.dropna(inplace=True)
        self.processed_data = df

    def prepare_sequences(self, test_size=0.2):
        """Normalizes data and splits into sequence windows for time-series forecasting."""
        # Features we will use for training
        features = ['Close', 'Volume', 'MA10', 'EMA12', 'RSI', 'MACD']
        data = self.processed_data[features].values
        target = self.processed_data[['Close']].values # Target is the closing price
        
        # Normalize data
        scaled_features = self.scaler_features.fit_transform(data)
        scaled_target = self.scaler_target.fit_transform(target)
        
        X, y = [], []
        # Create sequences of length `sequence_length`
        for i in range(len(scaled_features) - self.sequence_length):
            X.append(scaled_features[i:(i + self.sequence_length)])
            # The target is the Close price at the NEXT time step
            y.append(scaled_target[i + self.sequence_length])
            
        X = np.array(X)
        y = np.array(y)
        
        # Split into training and testing sets
        split_idx = int(len(X) * (1 - test_size))
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Keep dates for plotting the test set
        self.test_dates = self.processed_data.index[self.sequence_length + split_idx:]
        
        return X_train, y_train, X_test, y_test

# ==========================================
# 2. PyTorch Transformer Model Definition
# ==========================================
class PositionalEncoding(nn.Module):
    """
    Injects information about the relative or absolute position of the tokens 
    in the sequence. Essential for Transformers since they don't have recurrence.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to the input embeddings
        x = x + self.pe[:, :x.size(1), :]
        return x

class TimeSeriesTransformer(nn.Module):
    """
    A Transformer architecture adapted for multivariate time-series regression.
    """
    def __init__(self, num_features, d_model=64, n_heads=4, num_layers=3, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.d_model = d_model
        
        # Linear layer to project input features to the d_model dimension
        self.input_linear = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model*4, 
            dropout=dropout, 
            batch_first=True # Expects input shape: (batch, seq_len, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Fully connected layers to map transformer output to the final prediction
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1) # Output is a single value (Next Day Close Price)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, num_features)
        x = self.input_linear(x) # Project features
        x = self.pos_encoder(x)  # Add positional encoding
        
        # Pass through Transformer
        x = self.transformer_encoder(x)
        
        # Take the output of the last time step in the sequence to predict the next value
        x = x[:, -1, :] 
        
        # Pass through fully connected head
        out = self.fc(x)
        return out

# ==========================================
# 3. Model Trainer & Evaluator
# ==========================================
class ModelTrainer:
    """
    Handles the training loop, loss tracking, and evaluation of the PyTorch model.
    """
    def __init__(self, model, lr=0.001):
        # Use CUDA if available, else fallback to CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.model = model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        self.train_losses = []
        self.test_mses = []

    def train(self, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
        """Trains the Transformer model."""
        # Convert numpy arrays to PyTorch tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_test_t = torch.tensor(y_test, dtype=torch.float32).to(self.device)

        # Create DataLoader for batching
        train_data = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        print("Starting training...")
        for epoch in range(epochs):
            self.model.train()
            batch_losses = []
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                
                # Calculate loss
                loss = self.criterion(outputs, batch_y)
                batch_losses.append(loss.item())
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
            # Average training loss for the epoch
            epoch_loss = np.mean(batch_losses)
            self.train_losses.append(epoch_loss)
            
            # Evaluate on test set
            self.model.eval()
            with torch.no_grad():
                test_preds = self.model(X_test_t)
                test_mse = self.criterion(test_preds, y_test_t).item()
                self.test_mses.append(test_mse)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {epoch_loss:.6f} | Test MSE: {test_mse:.6f}")

    def predict(self, X_data):
        """Generates predictions for a given dataset."""
        self.model.eval()
        X_t = torch.tensor(X_data, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self.model(X_t).cpu().numpy()
        return predictions

# ==========================================
# 4. Visualization
# ==========================================
def plot_results(dates, actual, predicted, train_losses, test_mses, ticker):
    """Plots the actual vs predicted prices and the training loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Actual vs Predicted Stock Prices
    ax1.plot(dates, actual, label='Actual Price', color='blue')
    ax1.plot(dates, predicted, label='Predicted Price (Transformer)', color='red', linestyle='--')
    ax1.set_title(f'{ticker} Stock Price Prediction (Test Set)')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price (USD)')
    ax1.legend()
    ax1.grid(True)
    # Format x-axis dates nicely
    fig.autofmt_xdate(rotation=45)
    
    # Plot 2: Training Loss and Test MSE
    epochs_range = range(1, len(train_losses) + 1)
    ax2.plot(epochs_range, train_losses, label='Training Loss (MSE)', color='green')
    ax2.plot(epochs_range, test_mses, label='Test MSE', color='orange')
    ax2.set_title('Model Loss Over Epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Mean Squared Error')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 5. Main Execution Block
# ==========================================
if __name__ == "__main__":
    # --- Configuration ---
    TICKER = 'TXN'
    START_DATE = '2015-01-01'
    END_DATE = '2026-10-02'
    SEQ_LENGTH = 60 # Number of past days to look at to predict the next day
    EPOCHS = 60
    BATCH_SIZE = 32
    
    # 1. Process Data
    processor = StockDataProcessor(TICKER, START_DATE, END_DATE, SEQ_LENGTH)
    processor.fetch_data()
    processor.add_technical_indicators()
    X_train, y_train, X_test, y_test = processor.prepare_sequences()
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    num_features = X_train.shape[2]
    
    # 2. Initialize Model
    # d_model: Embedding dimension, n_heads: Attention heads, num_layers: Encoder layers
    transformer_model = TimeSeriesTransformer(
        num_features=num_features, 
        d_model=64, 
        n_heads=4, 
        num_layers=3, 
        dropout=0.1
    )
    
    # 3. Train Model
    trainer = ModelTrainer(transformer_model, lr=0.001)
    trainer.train(X_train, y_train, X_test, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    # 4. Predict and Inverse Transform
    predictions_scaled = trainer.predict(X_test)
    
    # Inverse transform to get actual dollar prices back
    predictions_actual = processor.scaler_target.inverse_transform(predictions_scaled)
    y_test_actual = processor.scaler_target.inverse_transform(y_test)
    
    # Calculate final Root Mean Squared Error (RMSE) on actual prices
    rmse = math.sqrt(mean_squared_error(y_test_actual, predictions_actual))
    print(f"\nFinal Test RMSE (in USD): ${rmse:.2f}")
    
    # 5. Plot Results
    plot_results(
        dates=processor.test_dates, 
        actual=y_test_actual, 
        predicted=predictions_actual, 
        train_losses=trainer.train_losses, 
        test_mses=trainer.test_mses,
        ticker=TICKER
    )