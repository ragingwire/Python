# Import libraries
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import LSTM, Linear, Module
from torch.cuda import is_available
import matplotlib.pyplot as plt

# Check CUDA availability
device = torch.device("cuda" if is_available() else "cpu")

# Define hyperparameters
learning_rate = 0.001
epochs = 100
input_size = 1  # Assuming we use only closing price
hidden_size = 64
num_layers = 1
prediction_length = 5

# Load and preprocess data
data = pd.read_csv("F:\downloads\TXN.csv")
closing_prices = data["Close"].values.reshape(-1, 1).astype(float)

# Normalize data (optional)
# scaler = MinMaxScaler(feature_range=(-1, 1))
# closing_prices = scaler.fit_transform(closing_prices)

# Create custom dataset class
class StockDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length - prediction_length

    def __getitem__(self, idx):
        sequence = self.data[idx : idx + self.seq_length]
        label = self.data[idx + self.seq_length : idx + self.seq_length + prediction_length]
        return sequence.to(device), label.to(device)

# Split data into training and testing sets
train_size = int(len(closing_prices) * 0.8)
train_data, test_data = closing_prices[:train_size], closing_prices[train_size:]

# Create datasets and dataloaders
seq_length = 30  # Define sequence length for prediction
train_dataset = StockDataset(train_data, seq_length)
test_dataset = StockDataset(test_data, seq_length)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the LSTM model
class LSTMModel(Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = LSTM(input_size, hidden_size, num_layers, batch_first=True).to(device)
        self.linear = Linear(hidden_size, output_size).to(device)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Get last hidden state
        x = self.linear(x)
        return x

# Initialize model and optimizer
model = LSTMModel(input_size, hidden_size, num_layers, prediction_length).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(epochs):
    for sequences, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = torch.nn.MSELoss()(outputs, labels.float())
        loss.backward()
        optimizer.step()

# Evaluate the model on test data and return predictions
def evaluate(dataloader):
    model.eval()
    with torch.no_grad():
        predictions = []
        for sequences, _ in dataloader:
            outputs = model(sequences)
            predictions.extend(outputs.cpu().numpy())
    return predictions

train_predictions = evaluate(train_loader)
test_predictions = evaluate(test_loader)

# Invert normalization (optional)
# if scaler:
#     train_predictions = scaler.inverse_transform(train_predictions)
#     test_predictions = scaler.inverse_transform(test_predictions)
#     closing_prices = scaler.inverse_transform(closing_prices)

# Visualize results
plt.figure(figsize=(12, 6))
plt.plot(closing_prices[: len(train_predictions)], label="Actual Prices (Train)")
plt.plot


