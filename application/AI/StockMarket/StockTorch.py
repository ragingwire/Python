import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from fontTools.misc.loggingTools import Timer
import time


class YFStockData :
    def __init__(self, ticker_symbol, start_date, end_date ):
        self.__data= []
        self.__ticker_symbol = ticker_symbol
        self.__start_date = start_date
        self.__end_date = end_date
        self.__data = yf.download( self.__ticker_symbol, start = self.__start_date, end = self.__end_date, auto_adjust = True )
        
    def getData ( self, category = 'Close' ):
        return self.__data [ category ]
    
    def setTickerSymbol (self, ticker_symbol ):
        self.__ticker_symbol = ticker_symbol
        
    def getTickerSymbol (self ):
        return self.__ticker_symbol
        
        

# Function to preprocess data for LSTM
def preprocess_data(data, seq_length):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    X, y = [], []
    scaled_data_length = len ( scaled_data )
    for i in range(seq_length, scaled_data_length ):
        to_append = scaled_data[i-seq_length:i, 0]
        X.append(to_append)
        element = scaled_data[i] 
        y.append( element )  # Changed to append the array
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, y, scaler


# Function to split data into training and testing sets
def split_data(X, y, test_size=0.2):
    """
    Split the data into training and testing sets based on test_size.
    """
    split = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    return X_train, X_test, y_train, y_test


class Timer :

    def __init__(self):
        """Initializes the timer."""
        self.__start_time = None
        self.__elapsed_time = None
        self.__loop_times = []

    def start( self ) :
        """Starts the timer."""
        if self.__start_time is not None:
            raise ValueError("Timer is already running. Call stop() first.")
        self.__start_time = time.perf_counter()

    def stop ( self ):
        """Stops the timer and records the elapsed time."""
        if self.__start_time is None:
            raise ValueError("Timer has not started. Call start() first.")
        self.__elapsed_time = time.perf_counter() - self.__start_time
        self.__start_time = None  # Reset start time for future use

    def elapsed ( self ):
        """Returns the elapsed time in seconds."""
        if self.__elapsed_time is None:
            if self._start_time is not None:
                return time.perf_counter() - self.__start_time #returns time for a running timer.
            raise ValueError("Timer has not been stopped yet or has not started.")

        return self.__elapsed_time

    def reset ( self ):
      """Resets the timer, clearing all recorded times."""
      self.__start_time = None
      self.__elapsed_time = None

    def add_loop_time ( self ):
        __loop_times.append ( elapsed () )
    
    def get_loops ( self ):
        return __loop_times.length ()
    
    def __enter__(self):
        """Allows the timer to be used as a context manager (with statement)."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stops the timer when exiting the context manager."""
        self.stop()
        if exc_type is not None:
            # Handle exceptions if needed.
            print(f"Timer exited with exception: {exc_type}, {exc_val}")
            return False  # Propagate exception
        return True  # Suppress exception



# Define the LSTM model class
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
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
def train_model(model, X_train, y_train, num_epochs=1250, learning_rate=0.001):
    """
    Train the LSTM model using the training data.
    """
    model.train ()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #for epoch in range(num_epochs):
    epoch = 0
    inputs = torch.from_numpy(X_train).float().to(device)
    labels = torch.from_numpy(y_train).float().to(device)
    while True:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        current_loss = loss.item ()
        if current_loss < 0.0003:
            print (f'Loss < 0.0003, ending model training, current_loss: {current_loss}')
            break
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
        epoch += 1

# Function to evaluate the model
def evaluate_model(model, X_test, y_test, scaler):
    """
    Evaluate the model on the test set and compute RMSE.
    Returns predictions and actual values in original scale.
    """
    model.eval()
    inputs = torch.from_numpy(X_test).float().to(device)
    with torch.no_grad():
        predictions = model(inputs).cpu().numpy()
        predictions = scaler.inverse_transform(predictions)
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
        rmse = np.sqrt(mean_squared_error(y_test_inv, predictions))
        print(f'RMSE: {rmse}')
        return predictions, y_test_inv

# Function to predict the next day's price
def predict_next_day(model, last_sequence, scaler):
    """
    Predict the stock price for the next day based on the last sequence.
    """
    model.eval()
    with torch.no_grad():
        input_seq = torch.from_numpy(last_sequence).float().to(device)
        prediction = model(input_seq).cpu().numpy()
        prediction = scaler.inverse_transform(prediction)
        return prediction[0][0]



STOCK_TICKER_SYMBOL = 'TXN'
STOCK_START_DATE = "2021-01-01"
STOCK_END_DATE = "2025-09-30"

RNN_NUM_INPUTS = 1
RNN_NUM_HIDDEN_SIZE = 50
RNN_NUM_LAYERS = 2

TRAINING_NUM_EPOCHS = 100
TRAINING_LEARNING_RATE = 0.02
TRAINING_SEQUENCE_LENGTH = 60

# Main execution
if __name__ == "__main__":
    # Parameters
    
    symbol = STOCK_TICKER_SYMBOL           # Stock symbol (e.g., Apple)
    start_date = STOCK_START_DATE # Start date for historical data
    end_date = STOCK_END_DATE   # End date for historical data
    seq_length = TRAINING_SEQUENCE_LENGTH           # Number of past days to use for prediction

    # Fetch and preprocess data
    stockData = YFStockData ( symbol, start_date, end_date )
    data = stockData.getData ()
    
    if len(data) < seq_length:
        print("Not enough data for the given sequence length.")
        exit()
    X, y, scaler = preprocess_data(data, seq_length)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Set device (CUDA if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for tensor calculation: {device}")

    # Initialize and train the model
    timer = Timer ()
    timer.start ()
    timer.stop ()
    timer.reset ()
    timer.start ()
    model = LSTMModel( RNN_NUM_INPUTS, RNN_NUM_HIDDEN_SIZE, RNN_NUM_LAYERS).to(device)
    train_model(model, X_train, y_train, TRAINING_NUM_EPOCHS, TRAINING_LEARNING_RATE)

    # Evaluate the model
    predictions, y_test_inv = evaluate_model(model, X_test, y_test, scaler)

    # Predict the next day's price
    last_sequence = X_test[-1].reshape(1, seq_length, 1)
    next_day_pred = predict_next_day(model, last_sequence, scaler)
    print(f"Predicted next day price: {next_day_pred}")
    timer.stop ()
    print(f"Elapsed calculation time [sec]: {timer.elapsed () }")
    

    # Visualize results
    plt.plot(y_test_inv, label="Actual Price")
    plt.plot(predictions, label="Predicted Price")
    plt.plot(next_day_pred, label="Next Day")
    plt.title(f"{symbol} Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
    print ( predictions )