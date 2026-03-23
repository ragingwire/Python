import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math
from datetime import datetime

# ==================== HYPERPARAMETERS AND CONFIG ====================
# These can be adjusted by the user for different performance
SEQ_LENGTH = 60          # Number of previous days used to predict the next day (look-back window)
BATCH_SIZE = 32
EPOCHS = 25              # Number of training epochs. Increase for better results (but longer training)
D_MODEL = 64             # Embedding dimension for the Transformer
NHEAD = 4                # Number of attention heads (must divide D_MODEL)
NUM_LAYERS = 2           # Number of Transformer encoder layers
DROPOUT = 0.1
LEARNING_RATE = 0.0005   # Learning rate for Adam optimizer

# Device configuration - uses CUDA (GPU) if available for faster training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# ==================== DATA DOWNLOAD AND PREPROCESSING ====================
# Download historical stock data from Yahoo Finance
stock_symbol = "AAPL"  # You can change this to any stock symbol like "GOOGL", "TSLA", etc.
print(f"Downloading historical stock data for {stock_symbol} from Yahoo Finance...")

# Download data (progress bar disabled for clean output)
data = yf.download(stock_symbol, start="2015-01-01", end=datetime.today().strftime("%Y-%m-%d"), progress=False)

# Use only the 'Close' price for univariate time series prediction
prices = data['Close'].values.reshape(-1, 1)

# Handle any potential missing values
prices = np.nan_to_num(prices)

print(f"Downloaded {len(prices)} days of data.")

# Scale the data to [0, 1] range for better neural network training
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices).flatten()  # Convert to 1D array

# ==================== CUSTOM DATASET CLASS ====================
class TimeSeriesDataset(Dataset):
    """
    Custom PyTorch Dataset for time series data.
    Creates sliding windows of SEQ_LENGTH past prices to predict the next price.
    """
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length
    
    def __len__(self):
        # Number of valid sequences
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        # Get input sequence and target value
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length]
        # Return as float32 tensors
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Split data into train (80%) and test (20%) - sequential split (no random shuffle for time series)
train_size = int(len(scaled_prices) * 0.8)
train_data = scaled_prices[:train_size]
test_data = scaled_prices[train_size:]

print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

# Create datasets and data loaders
train_dataset = TimeSeriesDataset(train_data, SEQ_LENGTH)
test_dataset = TimeSeriesDataset(test_data, SEQ_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==================== POSITIONAL ENCODING ====================
class PositionalEncoding(nn.Module):
    """
    Adds positional information to the input sequence so the Transformer knows the order of time steps.
    Uses sine and cosine functions as in the original 'Attention is All You Need' paper.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)  # Not a learnable parameter

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# ==================== TRANSFORMER MODEL ====================
class TransformerModel(nn.Module):
    """
    Transformer-based model for stock price prediction.
    - Projects 1D input to d_model dimension
    - Adds positional encoding
    - Uses Transformer Encoder layers (self-attention)
    - Predicts the next price using the last timestep output
    """
    def __init__(self, feature_size=1, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Linear layer to embed input features into higher dimension
        self.input_embedding = nn.Linear(feature_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu',
            batch_first=False,   # We will provide (seq_len, batch, features)
            norm_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final linear layer to output a single price prediction
        self.decoder = nn.Linear(d_model, 1)
        
    def forward(self, src):
        # src shape incoming: (batch_size, seq_length, 1)
        # 1. Embed input to d_model
        src = self.input_embedding(src)           # -> (batch_size, seq_length, d_model)
        
        # 2. Transpose for Transformer (seq_len first)
        src = src.transpose(0, 1)                 # -> (seq_length, batch_size, d_model)
        
        # 3. Add positional encoding
        src = self.pos_encoder(src)
        
        # 4. Pass through Transformer encoder
        output = self.transformer_encoder(src)    # -> (seq_length, batch_size, d_model)
        
        # 5. Use the last time step's output for next-step prediction
        output = output[-1, :, :]                 # -> (batch_size, d_model)
        
        # 6. Decode to scalar prediction
        output = self.decoder(output)             # -> (batch_size, 1)
        return output.squeeze(-1)                 # -> (batch_size,)

# Create model and move to GPU if available
model = TransformerModel(
    feature_size=1, 
    d_model=D_MODEL, 
    nhead=NHEAD, 
    num_layers=NUM_LAYERS, 
    dropout=DROPOUT
)
model = model.to(device)

print("Model architecture created successfully.")

# Loss function (MSE for regression) and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==================== TRAINING LOOP ====================
print("Starting model training...")
train_losses = []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    
    for batch_x, batch_y in train_loader:
        # Prepare data
        batch_x = batch_x.unsqueeze(-1).to(device)   # (batch, seq, 1)
        batch_y = batch_y.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_x)
        
        # Calculate loss
        loss = criterion(outputs, batch_y)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Training Loss: {avg_loss:.6f}")

print("Training completed!")

# ==================== EVALUATION AND PREDICTION ====================
print("Evaluating model on test data...")
model.eval()
test_predictions = []
actuals = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.unsqueeze(-1).to(device)
        outputs = model(batch_x)
        test_predictions.extend(outputs.cpu().numpy())
        actuals.extend(batch_y.numpy())

# Convert to numpy arrays
test_predictions = np.array(test_predictions)
actuals = np.array(actuals)

# Inverse scaling back to original price range
test_predictions = scaler.inverse_transform(test_predictions.reshape(-1, 1)).flatten()
actuals = scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()

# Calculate Mean Squared Error
mse = mean_squared_error(actuals, test_predictions)
print(f"\nTest Mean Squared Error (MSE): {mse:.4f}")

# ==================== PLOTTING RESULTS ====================
# Two separate figures as requested (loss in one, actual vs predicted in another)

# Figure 1: Training Loss Curve (this is the "loss" graph)
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss (MSE)', color='purple', linewidth=2)
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Figure 2: Actual vs Predicted Stock Prices (this is the "MSE" graph - MSE value shown in title)
plt.figure(figsize=(12, 6))
plt.plot(actuals, label='Actual Stock Price', color='blue', linewidth=2)
plt.plot(test_predictions, label='Predicted Stock Price', color='red', linestyle='--', linewidth=2)
plt.title(f'Actual vs Predicted Stock Prices for {stock_symbol}\nTest MSE: {mse:.2f}')
plt.xlabel('Time Steps (Test Period)')
plt.ylabel('Stock Price ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("All plots displayed. Code execution completed without errors.")