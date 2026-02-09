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
    

    def __init__ ( self, name, ticker, isin, wkn, numShares, distributionPolicy = _ETF_DISTRIBUTION_POLICY_ACCUMULATING, replication = _ETF_REPLICATION_PHYSICAL ):
        super.__init__( ticker, isin, wkn )
        self._numShares = numShares
        self._distributionPolicy = distributionPolicy
        self._replication = replication
        
    def getNumShares ( self ):
        return self._numShares
    
    def getDistributionPolicy ( self ) :
        return self._distributionPolicy
        
    def getReplication ( self ) :
        return self._replication


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
    
    APPLICATION_NAME = "ETF Tracker Application"
    APPLICATION_USAGE = "Usage: python your_script_name.py <input_directory_path>"
    
    def __init__ (self ):
        super ().__init__ ( ETFTrackerApplication.APPLICATION_NAME )
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
        print ( self.APPLICATION_NAME )
        
        return True
    


if __name__ == "__main__":
    
    etfTrackerApplication = ETFTrackerApplication ()
    
