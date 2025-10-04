import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import time

# --- 1. Setup and Environment Check ---
# --- Required Libraries: pip install yfinance scikit-learn matplotlib
print("--- PyTorch Transformer Stock Predictor ---")

# Determine device (ensures CUDA is used if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(42)
if device.type == 'cuda':
    torch.cuda.manual_seed(42)

# --- 2. Configuration Parameters ---
TICKER = "TXN"
START_DATE = "2020-01-01"
END_DATE = "2025-10-04"
LOOKBACK_WINDOW = 100  # Number of past days to look at to predict the next day
BATCH_SIZE = 32
EPOCHS = 500
LEARNING_RATE = 0.001
# Transformer parameters
D_MODEL = 64        # Dimension of the model's hidden layers
N_HEADS = 4         # Number of attention heads
N_LAYERS = 2        # Number of transformer encoder layers

# --- 3. Data Acquisition and Preprocessing ---

print(f"\nDownloading data for {TICKER}...")
data = yf.download(TICKER, start=START_DATE, end=END_DATE)
# Use only the 'Close' price for prediction
prices = data['Close'].values.reshape(-1, 1)

# Scale the data using MinMaxScaler (essential for neural networks)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

def create_sequences(data, seq_length):
    """
    Converts time-series data into sequences (X) and next-step targets (y).
    X = [Day t-N, ..., Day t-1]
    y = [Day t]
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_prices, LOOKBACK_WINDOW)
print(f"Created {X.shape[0]} sequences of length {LOOKBACK_WINDOW}.")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Convert numpy arrays to PyTorch Tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)

# Custom Dataset and DataLoader
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TimeSeriesDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 4. Transformer Model Definition ---

class TimeSeriesTransformer(nn.Module):
    """
    A simplified Transformer Encoder model for time series forecasting.
    It captures dependencies across the time steps in the input sequence.
    """
    def __init__(self, d_model, n_heads, n_layers, seq_len):
        super(TimeSeriesTransformer, self).__init__()
        
        # 1. Linear layer to embed the 1D price data into d_model dimension
        self.input_projection = nn.Linear(1, d_model)
        
        # 2. Positional Encoding (Crucial for time series in Transformers)
        # Allows the model to know the relative position of data points
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # Register as a fixed parameter

        # 3. Transformer Encoder Stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True  # Input shape: (batch, seq_len, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 4. Output Layer
        # Takes the output of the Transformer (which is seq_len x d_model)
        # and maps it to a single predicted price (1). We use the final
        # token's embedding for the prediction.
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, x):
        # x shape: (B, S, 1) -> (Batch, Sequence Length, Features)

        # 1. Project and add Positional Encoding
        x = self.input_projection(x) * np.sqrt(x.size(-1)) # Scale embedding
        x = x + self.pe[:, :x.size(1)] # Add positional encoding
        
        # 2. Pass through Transformer Encoder
        # Output shape: (B, S, d_model)
        encoder_output = self.transformer_encoder(x)
        
        # 3. Use the last time step's embedding for prediction
        # Shape: (B, d_model)
        final_token = encoder_output[:, -1, :] 

        # 4. Map to final output prediction (1 price value)
        # Shape: (B, 1)
        output = self.output_layer(final_token)
        return output

# Initialize Model, Loss, and Optimizer
model = TimeSeriesTransformer(
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    seq_len=LOOKBACK_WINDOW
).to(device)

criterion = nn.MSELoss() # Mean Squared Error is standard for regression
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"Model initialized. Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# --- 5. Training Loop ---

print("\nStarting model training...")
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        
        # Ensure data is contiguous and on the correct device
        X_batch = X_batch.contiguous().to(device)
        y_batch = y_batch.contiguous().to(device)

        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}")

end_time = time.time()
print(f"\nTraining finished in {end_time - start_time:.2f} seconds.")

# --- 6. Prediction and Visualization ---

# Set model to evaluation mode (essential for consistent behavior)
model.eval()

# Disable gradient calculation during inference (memory and speed optimization)
with torch.no_grad():
    # Make predictions on the test set
    predicted_scaled = model(X_test_t).cpu().numpy()
    
# Inverse transform predictions and actual values to the original price scale
predicted_prices = scaler.inverse_transform(predicted_scaled)
actual_prices = scaler.inverse_transform(y_test)

# Create the date indices for plotting
dates = data.index[-len(y_test):]

# --- Visualization ---

plt.figure(figsize=(14, 7), facecolor='#f5f5f5')
plt.style.use('seaborn-v0_8-whitegrid')

# Plot the actual closing prices
plt.plot(dates, actual_prices, label='Actual TXN Price', color='#1f77b4', linewidth=2)

# Plot the predicted prices
plt.plot(dates, predicted_prices, label='Predicted TXN Price', color='#ff7f0e', linestyle='--', linewidth=2)

plt.title(f'TXN Stock Price Prediction using PyTorch Transformer (Lookback={LOOKBACK_WINDOW})', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Closing Price (USD)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\n--- Prediction Complete ---")
print("Mean Absolute Error (Test Set):")
mae = np.mean(np.abs(predicted_prices - actual_prices))
print(f"{mae:.2f} USD")
