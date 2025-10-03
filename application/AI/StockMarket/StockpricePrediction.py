import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (device)

# Load your dataset
# For example, using a CSV file with 'Date' and 'Close' columns
#data = pd.read_csv('stock_prices.csv')
data = pd.read_csv("F:\\downloads\\TXN.csv")
data = data[['Date', 'Close']]
num_datapoints = data.size
data = data.loc[1000:, 'Date':'Close']
num_datapoints = data.size

# Preprocess the data
scaler = MinMaxScaler(feature_range=(-1, 1))
data['Close'] = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Convert data to sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 50
x, y = create_sequences(data['Close'].values, seq_length)

# Convert to PyTorch tensors
x = torch.from_numpy(x).float().to(device)
y = torch.from_numpy(y).float().to(device)

# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super(LSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size).to(device),
                            torch.zeros(1, 1, self.hidden_layer_size).to(device))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

model = LSTM().to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training the model
epochs = 30
for i in range(epochs):
    for seq, labels in zip(x, y):
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                             torch.zeros(1, 1, model.hidden_layer_size).to(device))

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    print (i)
    if i % 10 == 0:
        print(f'Epoch {i} loss: {single_loss.item()}')

# Making predictions
model.eval()
with torch.no_grad():
    test_inputs = data['Close'].values[-seq_length:].tolist()
    for _ in range(5):  # Predicting the next 30 days
        seq = torch.FloatTensor(test_inputs[-seq_length:]).to(device)
        with torch.no_grad():
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                                 torch.zeros(1, 1, model.hidden_layer_size).to(device))
            test_inputs.append(model(seq).item())

# Inverse transform the predictions
predicted_prices = scaler.inverse_transform(np.array(test_inputs[seq_length:]).reshape(-1, 1))
print(predicted_prices)
