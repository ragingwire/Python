import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt



class StockPricePredictor ( nn.Module ):
    def __init__(self, input_size=1, hidden_size=50, num_layers=3):
        """
        Initialize the LSTM model with specified input size, hidden size, and number of layers.
        """
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
        
    def forward(self, x):
        """
        Forward pass through the LSTM and fully connected layer.
        """
        # Initialize hidden and cell states
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        # Pass through LSTM
        out, _ = self.lstm(x, (h0, c0))
        # Pass the last timestep's output through the fully connected layer
        out = self.fc(out[:, -1, :])
        return out
    
    
    
        # Function to train the model
    def train ( self, X_train, y_train, num_epochs=1250, learning_rate=0.0005):
    
        """
        Train the LSTM model using the training data.
        """
        Super.train ()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam( parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            inputs = torch.from_numpy(X_train).float().to(device)
            labels = torch.from_numpy(y_train).float().to(device)
            optimizer.zero_grad()
            outputs = self (inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}')
                
                

    def evaluate ( self, X_test, y_test, scaler):
        """
        Evaluate the model on the test set and compute RMSE.
        Returns predictions and actual values in original scale.
        """
        eval()
        with torch.no_grad():
            inputs = torch.from_numpy(X_test).float().to(device)
            predictions =  self (inputs).cpu().numpy()
            predictions = scaler.inverse_transform(predictions)
            y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
            rmse = np.sqrt(mean_squared_error(y_test_inv, predictions))
            print(f'RMSE: {rmse}')
            return predictions, y_test_inv
    
    