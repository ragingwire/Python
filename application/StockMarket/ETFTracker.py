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


    
if __name__ == "__main__":
    ...