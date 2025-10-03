import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


data = pd.read_csv("F:\\downloads\\TXN.csv")

# Preprocess the data
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data = data['Close'].values.astype(float)

scaled_data = (data - np.min(data)) / (np.max(data) - np.min(data))
# Prepare data sequences
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        target = data[i+seq_length]
        sequences.append(seq)
        targets.append(target)
    return torch.FloatTensor(sequences).to(device), torch.FloatTensor(targets).to(device)

sequence_length = 60
X, y = create_sequences(scaled_data, sequence_length)

# Split the data
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size).to(device),
                            torch.zeros(1,1,self.hidden_layer_size).to(device))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]  # Return only the last output

model = LSTMModel().to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 50
for i in range(epochs):
    for seq, labels in zip(X_train, y_train):
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                             torch.zeros(1, 1, model.hidden_layer_size).to(device))
        
        y_pred = model(seq.unsqueeze(0))
        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i % 10 == 1:
        print(f'Epoch {i} loss: {single_loss.item()}')

# Prediction
model.eval()
train_predict = []
test_predict = []

for seq in X_train:
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                        torch.zeros(1, 1, model.hidden_layer_size).to(device))
        train_predict.append(model(seq.unsqueeze(0)).item())

for seq in X_test:
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                        torch.zeros(1, 1, model.hidden_layer_size).to(device))
        test_predict.append(model(seq.unsqueeze(0)).item())

# Inverse transform the predictions
train_predict = scaler.inverse_transform(np.array(train_predict).reshape(-1, 1))
test_predict = scaler.inverse_transform(np.array(test_predict).reshape(-1, 1))
y_train_real = scaler.inverse_transform(y_train.cpu().numpy().reshape(-1, 1))
y_test_real = scaler.inverse_transform(y_test.cpu().numpy().reshape(-1, 1))

# Plot results
plt.figure(figsize=(14,7))
plt.plot(data.index[sequence_length:], data['Price'][sequence_length:], label='Actual Price')
plt.plot(data.index[sequence_length:sequence_length+len(train_predict)], train_predict, label='Predicted - Train')
plt.plot(data.index[sequence_length+len(train_predict):], test_predict, label='Predicted - Test')
plt.legend()
plt.show()