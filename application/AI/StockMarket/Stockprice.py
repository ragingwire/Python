import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import math
import yfinance as yf
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torch import optim


class YFStockData ( object ):
    def __init__( self, ticker_symbol, start_date, end_date ):
        super ().__init__ ()
        self._data= []
        self._ticker_symbol = ticker_symbol
        self._start_date = start_date
        self._end_date = end_date
        self._download_failed = False
        self._data = yf.download( self._ticker_symbol, start = self._start_date, end = self._end_date, auto_adjust = True )
        self._data_length = len ( self._data )
        if ( self._data_length == 0 ) :
            self._download_failed = True

    def downloadFailed ( self ):
        return self._download_failed
    
    def getData ( self ):
        return self._data
    
    def setTickerSymbol (self, ticker_symbol ):
        self._ticker_symbol = ticker_symbol
        
    def getTickerSymbol (self ):
        return self._ticker_symbol
    
    def writeToCSV ( self, file_name ):
        self._data.to_csv ( file_name )
        
    def writeToExcel ( self, file_name, sheet_name ):
        self._data.to_excel ( file_name, sheet_name = self._ticker_symbol )
        

        
class StockPricePrediction ( object ):
    
    def __init__ ( self, stock_data = None, sequence_length = 5, batch_size = 16, epochs = 100, learning_rate = 0.001 ):
        super ().__init__ ()
        self._stock_data = stock_data
        self._sequence_length = sequence_length
        self._batch_size = batch_size
        self._epochs = epochs
        self._learning_rate = learning_rate
        self._model = None
        self._scaler = None
        self._train_loader = None
        self._test_loader = None
        self._train_size = 0
        self._test_size = 0
        self._train_data = None
        self._test_data = None
        self._predictions = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def __prepare_data__() :
        ...
        
    def __build_model__() :
        ...
    def __train_model__() :
        ...
        
    def __make_predictions__():
        ...
    def __plot_results__() :
        ...
        
        
        
if __name__ == "__main__":
    stockPricePrediction =  StockPricePrediction ()        
        
        
    def __prepare_data__ ( self ):
        data = self._stock_data['Close'].values.reshape(-1, 1)
        from sklearn.preprocessing import MinMaxScaler
        self._scaler = MinMaxScaler(feature_range=(0, 1))
        data_normalized = self._scaler.fit_transform(data)
        
        sequences = []
        targets = []
        
        for i in range(len(data_normalized) - self._sequence_length):
            sequences.append(data_normalized[i:i + self._sequence_length])
            targets.append(data_normalized[i + self._sequence_length])
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        train_size = int(len(sequences) * 0.8)
        
        train_sequences = sequences[:train_size]
        train_targets = targets[:train_size]
        
        test_sequences = sequences[train_size:]
        test_targets = targets[train_size:]
        
        self._train_size = len(train_sequences)
        self._test_size = len(test_sequences)
        
        train_dataset = TensorDataset(torch.tensor(train_sequences, dtype=torch.float32), torch.tensor(train_targets, dtype=torch.float32))
        test_dataset = TensorDataset(torch.tensor(test_sequences, dtype=torch.float32), torch.tensor(test_targets, dtype=torch.float32))
        
        self._train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True)
        self._        
        
        