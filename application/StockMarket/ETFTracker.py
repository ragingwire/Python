import numpy as np
import pandas as pd
import time
import math
import yfinance as yf



class Equity ( object ):
    def __init__( self, name, ticker, baseValue = 0.0, actualValue = 0.0, currency = "USD", isin = "", wkn = "" ):
        super ().__init__ ()
        self._name = name
        self._ticker = ticker
        self._isin = isin
        self._wkn = wkn
        self._actualValue = actualValue
        self._baseValue = baseValue
        self._currency = currency
        self.__getActualValue__ ()
    
    def getName ( self ):
        return self._name
    
    def getTicker ( self ) :
        return self._ticker 
    
    def getISIN ( self ):
        return self._isin
    
    def getWKN ( self ) :
        return self._wkn
    
    def __getActualValue__ ( self ) :
        ticker = yf.Ticker( self._ticker )
        self._actualValue = ticker.info.get('regularMarketPrice')
        self.__calculateGain__ ()
        return self._actualValue
    
    def getGain ( self ) :
        return self._gain
    
    def __updateActualValue__ ( self ):
        
        self.__caculateGain__ ()
        return True
    
    def __calculateGain__ ( self ) :
        self._gain = ( ( self._actualValue - self._baseValue ) / self._baseValue) * 100
        return self._gain
    

class ETF ( Equity ) :
    
    _ETF_DISTRIBUTION_POLICY_ACCUMULATING = "accumulating"
    _ETF_DISTRIBUTION_POLICY_DISTRIBUTING = "distributing"
    _ETF_REPLICATION_PHYSICAL = "physical"
    

    def __init__ ( self, name, ticker, baseValue, actualValue, numShares = 1, currency = "USD", isin = "", wkn = "", distributionPolicy = _ETF_DISTRIBUTION_POLICY_ACCUMULATING, replication = _ETF_REPLICATION_PHYSICAL ):
        super ().__init__( name, ticker, baseValue, actualValue, currency, isin, wkn )
        self._provider = ""
        self._hedged = False
        self._numShares = numShares
        self._distributionPolicy = distributionPolicy
        self._replication = replication
        self._totalValue = self._numShares * self._actualValue
        self._ter = 0.0
        
    
    def getTotalValue ( self ) :
        return self._totalValue
    
    


class ETFTracker ( object ) :
    
    _ETF_TRACKER_CONFIGURATION_FILE = "F:\downloads\ETF-Tracker.xlsx"
    
    def __init__ ( self, configFile = _ETF_TRACKER_CONFIGURATION_FILE ) : 
        super ().__init__ ()
        self._configFile = configFile
        self._dataFrame = pd.DataFrame ()
        self._etfs = []
        self.__readConfigFile__()
        
    def __readConfigFile__ ( self ):
        self._dataFrame = pd.read_excel ( self._ETF_TRACKER_CONFIGURATION_FILE )
        num_etfs = len ( self._dataFrame )
        for i in range ( num_etfs ) :
            _ = self._dataFrame.loc [ i : i ]
            etf = ETF ( _['Name'], _['Ticker'], _['BaseValue'], _['ActualValue'], _['NumShares'] )
            self._etfs.append ( etf )
            print ( etf )
        
        return True
    
    def __writeConfigFile__ ( self ):
        ...
                
        
        return True

class Application ( object ):
    
    def __init__ (self, applicationName ):
        super ().__init__ ()
        self.applicationName = applicationName
        self.commandLineArgs = []
        self.options = []
        self.numCommandLineArgs = 0
        
    def getApplicationName ( self ) :
        return self.ApplicationName
    
    def getNumCommandLineArgs (self):
        return self.numCommandLineArgs
    
    def printUsage ( self ):
        pass
    
    def log (self, logstr ):
        print ( logstr )
    
    def __run__ (self ):
        pass
    
    def exit (self,exitCode = 0 ):
        self.exitCode = exitCode
        sys.exit ( self.exitCode )
        
    def __handeCommandLineArgs__ ( self ):
        pass

class ETFTrackerApplication ( Application ):
    
    _APPLICATION_NAME = "ETF Tracker Application"
    _APPLICATION_USAGE = "Usage: python etftracker.py <configfile>"
    
    def __init__ (self ):
        super ().__init__ ( ETFTrackerApplication._APPLICATION_NAME )
        self.__run__ ()
    
    def __handleCommanddLineArgs__ ( self ):
        self.numCommandLineArgs = len ( sys.argv )
        if self.numCommandLineArgs == 1:
            self.__printUsage__ ()
            return False
        elif self.numCommandLineArgs == 2:
            ...
        elif self.numCommandLineArgs == 3:
            ...
        
        return True
    
    def __printUsage__ (self ):
        return True
    
    def __run__ ( self ):
        print ( self._APPLICATION_NAME )
        etfTracker = ETFTracker ()
        
        return True
    


if __name__ == "__main__":
    
    etfTrackerApplication = ETFTrackerApplication ()
    
