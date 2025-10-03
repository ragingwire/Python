'''
Created on 28 Apr 2023

@author: Ragingwire
'''

import pandas as pd
from yahoo_fin import stock_info as si


STOCK_MARKET_SOURCE_YAHOO = 'Yahoo'
STOCK_MARKET_DEFAULT_CVS_PATH = 'C://temp'
STOCK_MARKET_DEFAULT_DATA_FORMAT = 'csv'
STOCK_MARKET_DATA_NO_ERROR = 0
STOCK_MARKET_DATA_LOAD_ERROR = 1
STOCK_MARKET_DATA_WRITE_ERROR = 2
STOCK_MARKET_DATA_DATE_COLUMN = "date"
STOCK_MARKET_DATA_FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]

class StockMarketException (Exception):

    def __init__ (self, errorCode = STOCK_MARKET_DATA_NO_ERROR ):
        self._exception = errorCode
    
    def getException ( self ):
        return self._exception


class StockMarketData ( object ):
    
    def __init__ (self, ticker, path = STOCK_MARKET_DEFAULT_CVS_PATH, source= STOCK_MARKET_SOURCE_YAHOO ):
        self._ticker = ticker
        self._path = path
        self._stockMarketSource = source
        self._df = pd.DataFrame ()
        self._errorCode = STOCK_MARKET_DATA_NO_ERROR

        
    def getErrorCode ( self ):
        return self._errorCode
    
    def setTicker (self, ticker):
        self._ticker = ticker
        
    def getTicker (self):
        return self._ticker
    
    def setPath (self, path):
        self._path = path
    
    def getPath (self):
        return self._path
    
    def getStockMarketSource (self ):
        return self._stockMarketSource
    
    def getMarketData (self, source = STOCK_MARKET_SOURCE_YAHOO, fileFormat = STOCK_MARKET_DEFAULT_DATA_FORMAT ):
        try:
            self._df = si.get_data( self.getTicker () )
            self.df.dropna(inplace=True)
            self.__writeToFile ( fileFormat )
            self._errorCode = STOCK_MARKET_DATA_WRITE_ERROR
        except:
            self._errorCode = STOCK_MARKET_DATA_LOAD_ERROR
            #raise StockMarketException ( self._errorCode )
        if STOCK_MARKET_DATA_DATE_COLUMN not in self._df.columns:
            self._df[STOCK_MARKET_DATA_DATE_COLUMN] = self._df.index
        return self._df
     
    def __writeToFile (self, fileFormat = STOCK_MARKET_DEFAULT_DATA_FORMAT ):
        fullPathName = self.getPath () + "//" + self._ticker
        csvFileName = fullPathName + '.csv'
        xlsFileName = fullPathName + '.xlsx'
        if fileFormat == STOCK_MARKET_DEFAULT_DATA_FORMAT:
            self._df.to_csv ( csvFileName )
            self._df.to_excel ( xlsFileName )
            