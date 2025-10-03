import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# CUDA support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load stock data (replace with your data source)
data = pd.read_csv("F:\\downloads\\TXN.csv")
data = data['Close'].values

# Preprocess data
# Normalize the data
data = (data - np.min(data)) / (np.max(data) - np.min(data))

# Create input and target sequences
sequence_length = 10
input_seq = []
target_seq = []
for i in range(len(data) - sequence_length):
    input_seq.append(data[i:i+sequence_length])
    target_seq.append(data[i+sequence_length])

input_seq = np.array(input_seq)
target_seq = np.array(target_seq)

# Convert to PyTorch tensors
input_seq = torch.Tensor(input_seq).to(device)
target_seq = torch.Tensor(target_seq).to(device)



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


    
model = LSTMModel(sequence_length, 50, 2, 1).to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for i in range(len(input_seq)):
        input_data = input_seq[i].unsqueeze(1)
        target_data = target_seq[i].unsqueeze(0)

        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, target_data)
        loss.backward()
        optimizer.step()
        #print (i)

    print (epoch)
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

# Prediction
test_data = input_seq[-1].unsqueeze(0)
predicted_value = model(test_data).item()
print("Predicted value:", predicted_value)