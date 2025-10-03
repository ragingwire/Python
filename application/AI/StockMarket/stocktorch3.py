import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn, optim

# Check if CUDA is available and set PyTorch to use GPU or CPU accordingly
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print ( device )

# replace 'your_file.csv' with the path to your CSV file
data = pd.read_csv("F:\downloads\TXN.csv")
print (data)

# prepare the data
data = data['Close'].values
data = data.astype('float32')
data = np.reshape(data, (-1, 1))

# split the data into train and test sets
train_size = int(len(data) * 0.80)
test_size = len(data) - train_size
train, test = data[0:train_size,:], data[train_size:len(data),:]

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# convert to PyTorch tensors and send to the device
trainX = torch.from_numpy(trainX).to(device)
trainY = torch.from_numpy(trainY).to(device)
testX = torch.from_numpy(testX).to(device)
testY = torch.from_numpy(testY).to(device)

# create the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size).to(device), torch.zeros(1,1,self.hidden_layer_size).to(device))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

model = LSTM().to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train the model
epochs =1000
for i in range(epochs):
    model.train()
    optimizer.zero_grad()
    model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device), torch.zeros(1, 1, model.hidden_layer_size).to(device))
    y_pred = model(trainX)
    single_loss = loss_function(y_pred, trainY)
    single_loss.backward()
    optimizer.step()

    if i%25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

# make predictions
model.eval()
trainPredict = model(trainX)
testPredict = model(testX)

# plot baseline and predictions
plt.plot(data)
plt.plot(trainPredict.data.cpu().numpy())
plt.plot(testPredict.data.cpu().numpy())
plt.show()
