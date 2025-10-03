import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load historical stock price data
data = pd.read_csv("F:\\downloads\\TXN.csv")

# Preprocess the data
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data = data['Close'].values.astype(float)

# Normalize the data
data = (data - np.min(data)) / (np.max(data) - np.min(data))
scale_factor = np.min(data) / ((np.max(data) - np.min(data)))

# Split the data into input (X) and target (y) sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 30
X, y = create_sequences(data, seq_length)

# Convert data to PyTorch tensors
X = torch.tensor(X).float().to(device)
y = torch.tensor(y).float().to(device)
print ( X.shape )
print (y.shape )

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

# Instantiate the model
input_size = 1
hidden_size = 64
num_layers = 2
output_size = 1
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X.unsqueeze(2))
    loss = criterion(output.squeeze(), y)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        print(f'Epoch [{epoch+1}/{epochs}]')

# Predict the next 5 days of stock prices

#test_data = X[-1].unsqueeze(0)
#predicted_value = model(test_data).item()

prediction_days = 5
model.eval()
with torch.no_grad():
    future_seq = X[-1:].clone()
    predictions = []
    print (future_seq.shape )
    print (future_seq )
    for _ in range( prediction_days ):
        future_seq = model(future_seq.unsqueeze(2))
        predictions.append(future_seq.squeeze().item())
        print (future_seq)
        ten1 = future_seq[0][1:]
        print (ten1 )
        ten2 = future_seq[0][-1:]
        print (ten2)
        future_seq = torch.cat((ten1, ten2)).unsqueeze(0)
        print (future_seq )
        print (predictions )

# Denormalize the predictions
print (predictions )
predictions = np.array(predictions) * (np.max(data) - np.min(data)) + np.min(data)
print (predictions)

# Visualize the results
plt.figure(figsize=(10, 5))
plt.plot(np.arange(len(data)), data, label='Historical Data')
plt.plot(np.arange(len(data), len(data)+5), predictions, label='Predictions', marker='o')
plt.xlabel('Day')
plt.ylabel('Stock Price')
plt.title('Stock Price Prediction with LSTM')
plt.legend()
plt.show()