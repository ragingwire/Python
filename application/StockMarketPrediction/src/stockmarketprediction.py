'''
Created on 28 Apr 2023

@author: Ragingwire
'''

import pandas as pd
import numpy as np
from sklearn import preprocessing
from stockmarket import StockMarketData
import mplfinance as mpf
import time
from IPython.core.pylabtools import figsize
import matplotlib.pyplot as plt



class FeatureEngineering ( object ):
    
    def __init__ (self, pandasDf ):
          self._pandasDf = pandasDf
          self._trainingData = pd.DataFrame ()
          self._testData = pd.DataFrame ()
          self._numDataSets = len ( self._pandasDf )
          self._numTestSets = 0
          self._numTrainingSets = 0;
          self._dateNow = time.strftime("%Y-%m-%d")
    
    def __shuffleAndSplit (self, testRatio = 0.2 ):
        shuffledIndices = np.random.permutation ( self._numDataSets )
        self._numTestSets = int ( self._numDataSets * testRatio )
        testIndices = shuffledIndices [ : self._numTestSets ]
        trainIndices = shuffledIndices [ self._numTestSets : ]
        self._trainingData = self._pandasDf.iloc [ trainIndices ]
        self._testData = self._pandasDf.iloc [ testIndices ]
        
    def __scaleData (self, feature_columns=['close','adjclose', 'volume', 'open', 'high', 'low']):
        column_scaler = {}
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler( feature_range = (-1, 1) )
            self._pandasDf[column] = scaler.fit_transform (np.expand_dims( self._pandasDf [column].values, axis=1))
            column_scaler[column] = scaler
        return
    
    def engineer (self, feature_columns=['adjclose', 'volume', 'open', 'high', 'low'], lookupStep=1):
        self.__shuffleAndSplit ()   
        self.__scaleData ()
        training_set = self._pandasDf.iloc[:800, 1:2].values
        test_set = self._pandasDf.iloc[800:, 1:2].values
        #self._pandasDf ['future'] = self._pandasDf ['adjclose'].shift(-lookupStep)
        #last_sequence = np.array(self._pandasDf [feature_columns].tail(-lookupStep))
        #self._pandasDf.dropna(inplace=True)
        new_dataset=pd.DataFrame(index=range(0,len(self._pandasDf)),columns=['date','close'])
        numdata = len ( self._pandasDf )
        for i in range(0,numdata):
            new_dataset["date"][i]=self._pandasDf ["date"][i]
            new_dataset["close"][i]=self._pandasDf ["close"][i]
            
        final_dataset = new_dataset.values
        return
    

    
    def getNumDataSets (self):
        return self.numDataSets
        
        
    

stockMarket = StockMarketData ( 'TXN', 'F://downloads' )

print ( stockMarket.getTicker () )
print ( stockMarket.getPath () )
print ( stockMarket.getStockMarketSource() )
pdf = stockMarket.getMarketData()


print (pdf.values )

pdf.hist ( bins = 50, figsize = (12, 8))
plt.show ()

featureEngineering = FeatureEngineering ( pdf )
featureEngineering.engineer ()

if pdf.empty:
    print ("Error loading market data for ticker symbol: " + stockMarket.getTicker () )
else:
    print ( pdf.tail( 10 ) )
    print ( pdf.info () )
    mpf.plot( pdf,type='line',mav=(3,6,9),volume=False,show_nontrading=False, title=stockMarket.getTicker () )
    
    
    
    
    
    
    