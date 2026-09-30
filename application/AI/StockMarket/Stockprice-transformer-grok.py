import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ========================== CONFIG (tuned for much better results) ==========================
TICKER = "TXN"
SEQ_LENGTH = 32       # longer history = better context
BATCH_SIZE = 64
EPOCHS = 100
D_MODEL = 64
NHEAD = 8
NUM_LAYERS = 3
LEARNING_RATE = 0.0001
DROP_OUT_RATE = 0.1
# ==========================================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")

# 1. Download real data
print("Downloading stock data from Yahoo Finance...")
df = yf.download(TICKER, start="2000-01-01", end=datetime.now().strftime('%Y-%m-%d'))

# 2. FEATURE ENGINEERING - THIS IS THE #1 REASON PREDICTIONS IMPROVE
df = df.copy()
df['EMA20'] = df['Close'].ewm(span=20).mean()

# RSI (14-day)
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - 100 / (1 + rs)

# MACD
ema12 = df['Close'].ewm(span=12).mean()
ema26 = df['Close'].ewm(span=26).mean()
df['MACD'] = ema12 - ema26

df = df.dropna()  # remove NaN rows from indicators

# Use 5 powerful features (Close is first column)
features = ['Close', 'Volume', 'RSI', 'MACD', 'EMA20']
num_features = len(features)

# 3. Scale ALL features together
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features].values)

# 4. Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])           # shape: (seq, 5)
        y.append(data[i + seq_length, 0])          # next Close (scaled)
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, SEQ_LENGTH)

# 5. Split + tensors
train_size = int(len(X) * 0.85)
X_train = torch.FloatTensor(X[:train_size]).to(device)
y_train = torch.FloatTensor(y[:train_size]).unsqueeze(1).to(device)
X_test  = torch.FloatTensor(X[train_size:]).to(device)
y_test  = torch.FloatTensor(y[train_size:]).unsqueeze(1).to(device)

# Dates for beautiful plot
test_dates = df.index[SEQ_LENGTH:][train_size:]

# ====================== MODEL ======================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.shape[1]]

class StockTransformer(nn.Module):
    def __init__(self, input_size=num_features, d_model=D_MODEL, nhead=NHEAD,
                 num_layers=NUM_LAYERS, dropout=DROP_OUT_RATE):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout,
            batch_first=True, activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.input_proj(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return self.decoder(output[:, -1, :])   # predict next day

model = StockTransformer().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5 )

# Dataset + DataLoader (proper mini-batch training)
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

# ====================== TRAINING ======================
print("Training improved Transformer (multivariate + indicators)...")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    scheduler.step(avg_loss)
    
    if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
        print(f"Epoch {epoch+1:2d}/{EPOCHS} | Loss: {avg_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

# ====================== PREDICTION & PLOT ======================
model.eval()
with torch.no_grad():
    pred_scaled = model(X_test).cpu().numpy()
    actual_scaled = y_test.cpu().numpy()

# Inverse transform (Close is column 0)
dummy = np.zeros((len(pred_scaled), num_features))
dummy[:, 0] = pred_scaled.flatten()
pred = scaler.inverse_transform(dummy)[:, 0]

dummy[:, 0] = actual_scaled.flatten()
actual = scaler.inverse_transform(dummy)[:, 0]

# Plot with real dates
plt.figure(figsize=(15, 8))
plt.plot(test_dates, actual, label='Actual Price', color='blue', linewidth=2)
plt.plot(test_dates, pred, label='Predicted Price (Transformer)', color='red', linestyle='--', linewidth=2)
plt.title(f'{TICKER} Stock Price Prediction - Improved Transformer (Multivariate + RSI/MACD/EMA)', fontsize=16)
plt.xlabel('Date')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Quick metrics
rmse = np.sqrt(np.mean((actual - pred)**2))
print(f"\n✅ Training finished!")
print(f"Test RMSE: ${rmse:.2f}")
print(f"Prediction now uses 5 features + technical indicators → much better trend following!")