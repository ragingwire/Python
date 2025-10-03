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