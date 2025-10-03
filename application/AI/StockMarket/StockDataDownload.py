import numpy as np
import pandas as pd
import yfinance as yf
import sys
import matplotlib.pyplot as plt
from yfinance.utils import get_ticker_by_isin
from datetime import datetime
import time


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
    
    def __getData__ ( self, category = 'Close' ):
        return self.__data [ category ]
    
    def setTickerSymbol (self, ticker_symbol ):
        self._ticker_symbol = ticker_symbol
        
    def getTickerSymbol (self ):
        return self.__ticker_symbol
    
    def writeToCSV ( self, file_name ):
        self._data.to_csv ( file_name )
        
    def writeToExcel ( self, file_name, sheet_name ):
        self._data.to_excel ( file_name, sheet_name = self._ticker_symbol )
        

class StockDownloadApplication ( object ):
    
    _APPLICATION_NAME = "Stock Data Downloader"
    _APPLICATION_USAGE = "Usage: python StockDataDownlaod.py <ticker>"
    _OUTPUT_DIRECTORY = "F:\\development\\Python\\application\\Data\\"
    _EXTENSION_CSV = ".csv"
    _EXTENSION_XLSX = ".xlsx"
    
    def __init__ ( self ):
        super ().__init__ ()
        self._ticker_symbol = ""
        self._start_date = datetime(2020, 1, 1)
        self._end_date = time.strftime("%Y-%m-%d")
        self._stockData = None 
        self.__run__()
        
    
    def __log__ ( self, message ):
        print ( message )
        
    def __run__ (self) :
        self.__log__ ( self._APPLICATION_NAME )
        self.__processCmdLnArgs__()
        self.__getStockData__ ()
        self.__plotStockData__ ()
        
    def __getStockData__ (self ):
        try:
            self._stockData = YFStockData ( self._ticker_symbol, self._start_date, self._end_date )
        except Exception as e: 
            pass
        if self._stockData.downloadFailed () :
            self.__log__ ( 'failed to download stock data for stock symbol: ' + self._ticker_symbol )
            self.__exit__( 1 )  
        try :
            self._stockData.writeToCSV ( self._OUTPUT_DIRECTORY + self._ticker_symbol + self._EXTENSION_CSV )
            self.__log__ ( 'stock data written to ' + self._OUTPUT_DIRECTORY + self._ticker_symbol + self._EXTENSION_CSV )
            self._stockData.writeToExcel ( self._OUTPUT_DIRECTORY + self._ticker_symbol + self._EXTENSION_XLSX, self._ticker_symbol ) 
            self.__log__ ( 'stock data written to ' + self._OUTPUT_DIRECTORY + self._ticker_symbol + self._EXTENSION_XLSX )
        except Exception as e: 
            self.__log__ ( 'error writing stock data to file ' + str(e) )
            self.__exit__( 1 )
    
    def __plotStockData__ ( self ):
        data = self._stockData._data
        plot_title = self._ticker_symbol + ' stock price'
        self.__log__ ( 'plotting stock data for ' + self._ticker_symbol )
        plt.figure(figsize=(10, 6))
        plt.plot(data.index, data['Close'], label='Close Price')
        plt.title(plot_title)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid()
        plt.show()


    def __processCmdLnArgs__ ( self ) :
        if len ( sys.argv ) < 2:
            self.__log__ ('stock ticker symbol is missing')
            self.__exit__( 1 )
        self._ticker_symbol = sys.argv [1]
        
    def __exit__ (self, exit_code = 0 ):
        sys.exit ( exit_code )
    
    

if __name__ == "__main__":
    stockDownloadApplication =  StockDownloadApplication ()
        