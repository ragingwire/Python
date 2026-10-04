import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import math
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from datetime import datetime


class YFStockHistory ( object ):
    def __init__( self, ticker_symbol, start_date, end_date ):
        super ().__init__ ()
        self.__data= []
        self.__ticker_symbol = ticker_symbol
        self.__start_date = start_date
        self.__end_date = end_date
        self.__download_failed = False
        self.__download_stock_data ()
        self.__num_data_points = len ( self.__data )
        if ( self.__num_data_points == 0 ) :
            self.__download_failed = True

        
    def __download_stock_data (self ):
        
        try:
            self.__data = yf.download ( self.__ticker_symbol, start = self.__start_date, end = self.__end_date, auto_adjust = True )
            if isinstance( self.__data.columns, pd.MultiIndex ): 
                    self.__data.columns = self.__data.columns.get_level_values ( 0 )
        except Exception as e: 
            self.__download_failed = True
        
    
    def download_failed ( self ):
        return self.__download_failed
    
    def get_stock_data ( self ):
        return self.__data
    
    def set_ticker_symbol (self, ticker_symbol ):
        self.__ticker_symbol = ticker_symbol
        
    def get_ticker_symbol (self ):
        return self.__ticker_symbol
    
    def write_to_csv ( self, file_name ):
        self.__data.to_csv ( file_name )
        
    def write_to_excel ( self, file_name, sheet_name ):
        self.__data.to_excel ( file_name, sheet_name = self.__ticker_symbol )
        
        
        
        
class PositionalEncoding ( nn.Module ):
    
    def __init__( self, d_model, dropout=0.1, max_len=1000 ):
        super().__init__()
        self.dropout = nn.Dropout( p=dropout )
        
        # Create positional encoding matrix
        position = torch.arange( 0, max_len, dtype=torch.float ).unsqueeze(1)
        div_term = torch.exp( torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)  # Not a learnable parameter

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
        
        
class StockPricePredictionTransformer ( object ):
    
    __SPP_TICKER = 'TXN'
    __SPP_START_DATE = start_date = datetime(2020, 1, 1)
    __SPP_END_DATE = time.strftime("%Y-%m-%d")
    __SPP_SEQ_LENGTH = 30
    __SPP_EPOCHS = 100
    __SPP_BATCH_SIZE = 16
    __SPP_LEARNING_RATE = 0.001          # Number of training epochs. Increase for better results (but longer training)
    __SPP_D_MODEL = 64                   # Embedding dimension for the Transformer
    __SPP_NHEAD = 4                      # Number of attention heads (must divide D_MODEL)
    __SPP_NUM_LAYERS = 4  
    __SPP_CUDA_DEVICE_GPU = 'cuda'
    __SPP_CUDA_DEVICE_CPU = 'cpu'
    
    
    def __init__ ( self, ticker = "",  sequence_length = 5, batch_size = 16, epochs = 100, learning_rate = 0.001 ):
        super ().__init__ ()
        self.__ticker = ticker
        self.__sequence_length = sequence_length
        self.__batch_size = batch_size
        self.__epochs = epochs
        self.__learning_rate = learning_rate
        self.__model = None
        self.__scaler = None
        self.__train_loader = None
        self.__test_loader = None
        self.__train_size = 0
        self.__test_size = 0
        self.__raw_data = None
        self.__scaled_data = None
        self.__train_data = None
        self.__test_data = None
        self.__predictions = None
        self.__device = torch.device( self.__SPP_CUDA_DEVICE_GPU if torch.cuda.is_available() else self.__SPP_CUDA_DEVICE_CPU )
        self.__get_stock_data ()
        self.__feature_engineering ()
        self.__scale_data ( "forward" )
        self.__prepare_data ()
        
        
    def __get_stock_data (self ) :
        self.__raw_data = YFStockHistory ( self.__SPP_TICKER, self.__SPP_START_DATE, self.__SPP_END_DATE  ).get_stock_data ()
        
    def __feature_engineering ( self ) :
        df = self.__raw_data
        df["Return"]    = df["Close"].pct_change()
        df["MA10"]      = df["Close"].rolling(10).mean()
        df["MA30"]      = df["Close"].rolling(30).mean()
        df["EMA12"]     = df["Close"].ewm(span=12, adjust=False).mean()
        df["EMA26"]     = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"]      = df["EMA12"] - df["EMA26"]
        delta           = df["Close"].diff()
        gain            = delta.clip(lower=0).rolling(14).mean()
        loss            = (-delta.clip(upper=0)).rolling(14).mean()
        rs              = gain / (loss + 1e-9)
        df["RSI"]       = 100 - 100 / (1 + rs)
        df["BB_mid"]    = df["Close"].rolling(20).mean()
        df["BB_std"]    = df["Close"].rolling(20).std()
        df["BB_upper"]  = df["BB_mid"] + 2 * df["BB_std"]
        df["BB_lower"]  = df["BB_mid"] - 2 * df["BB_std"]
        df["BB_width"]  = (df["BB_upper"] - df["BB_lower"]) / (df["BB_mid"] + 1e-9)
        df["Vol_MA10"]  = df["Volume"].rolling(10).mean()
        df["OBV"]       = (np.sign(df["Close"].diff()) * df["Volume"]).cumsum()
        df.dropna(inplace=True)
    
    def __scale_data (self, direction ):
        scaler = MinMaxScaler()
        
        if direction == "forward" :
            self.__scaled_data = scaler.fit_transform ( self.__raw_data )
        else :
            ...        
    
    def __prepare_data( self ) :
        X, y = [], []
        for i in range(len( self.__scaled_data ) - self.__SPP_SEQ_LENGTH ):
            X.append( self.__scaled_data [i:i + self.__SPP_SEQ_LENGTH ] )
            y.append( self.__scaled_data [i + self.__SPP_SEQ_LENGTH ])  # next day's close

            #X = np.array(X)
            #y = np.array(y)
        ...
        
    def __build_model( self ) :
        ...
    def __train_model( self ) :
        ...
        
    def __make_predictions( self ):
        ...
    def __plot_results( self ) :
        ...
        


        
if __name__ == "__main__":
    StockPricePredictionTransformer ( )
    