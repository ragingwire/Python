import numpy as np
import pandas as pd
import time
import math
import yfinance as yf

class Equity ( object ):
    def __init__( self, name, ticker, isin, wkn ):
        super ().__init__ ()
        self._name = name
        self._ticker = ticker
        self._isin = isin
        self._wkn = wkn
    
    def getName ( self ):
        return self._name
    
    def getTicker ( self ) :
        return self._ticker 
    
    def getISIN ( self ):
        return self._isin
    
    def getWKN ( self ) :
        return self._wkn
    

class ETF ( Equity ) :
    
    _ETF_DISTRIBUTION_POLICY_ACCUMULATING = "accumulating"
    _ETF_DISTRIBUTION_POLICY_DISTRIBUTING = "distributing"
    _ETF_REPLICATION_PHYSICAL = "physical"
    

    def __init__ ( self, distributionPolicy = _ETF_DISTRIBUTION_POLICY_ACCUMULATING, replication = _ETF_REPLICATION_PHYSICAL ):
        super.__init__( ticker, isin, wkn )
        self._fundProvider = ""
        self._etfName = ""
        self._ticker = ""
        self._isin = ""
        self._wkn = ""
        self._fundCurrency = "USD"
        self._hedged = False
        self._numShares = 0
        self._distributionPolicy = distributionPolicy
        self._replication = replication
        self._actualPrice = 0.0
        self._basePrice = 0.0
        self._actualPrice = 0.0
        self._totalValue = 0.0
        self._gain = 0.0
        self._ter = 0.0
        
    def getNumShares ( self ):
        return self._numShares
    
    def getDistributionPolicy ( self ) :
        return self._distributionPolicy
        
    def getReplication ( self ) :
        return self._replication


class ETFTracker ( object ) :
    
    _ETF_TRACKER_CONFIGURATION_FILE = "F:\downloads\ETF-Tracker.xlsx"
    
    def __init__ ( self, configFile = _ETF_TRACKER_CONFIGURATION_FILE ) : 
        super.__init__ ()
        self.configFile = configFile
        self.dataFrame = pd.DataFrame ()
        self.etfs = []
        
    def __readConfigFile__ ( self ):
        self.dataFrame = pd.read_excel ( self._ETF_TRACKER_CONFIGURATION_FILE )
        
        return True
    
    def __writeConfigFile__ ( self ):
        self.dataFrame = pd.write_excel ( self._ETF_TRACKER_CONFIGURATION_FILE )
        num_etfs = len ( self.dataFrame )
        for etf in range ( num_etf ):
            
        
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
    _APPLICATION_USAGE = "Usage: python your_script_name.py <input_directory_path>"
    
    def __init__ (self ):
        super ().__init__ ( ETFTrackerApplication._APPLICATION_NAME )
        ...
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
        
        return True
    


if __name__ == "__main__":
    
    etfTrackerApplication = ETFTrackerApplication ()
    
