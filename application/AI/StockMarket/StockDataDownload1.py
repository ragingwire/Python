from datetime import datetime
import yfinance as yf
import mplfinance as mpf
import time
import sys


if len ( sys.argv ) < 2:
    print ('stock ticker symbol is missing')
    exit( 1 )

ticker_symbol = '^GDAXI'
cvs_data_name = 'F:\\downloads\\stock.csv'
xls_data_name = 'F:\\downloads\\stock.xlsx'
xls_sheet_name = 'stockprice'

start_date = datetime(2023, 1, 1)
end_date = time.strftime("%Y-%m-%d")

ticker_symbol = sys.argv [1]
cvs_data_name = 'F:\\downloads\\' + ticker_symbol + '.csv'
xls_data_name = 'F:\\downloads\\' + ticker_symbol + '.xlsx'
xls_sheet_name = ticker_symbol
plot_title = ticker_symbol + ' stock price'

data = yf.download( ticker_symbol, start=start_date, end=end_date,multi_level_index=False)
print ( data )

mpf.plot(data,type='line',mav=(3,6,9),volume=True,show_nontrading=True, title=plot_title )

print ( 'writing ' + cvs_data_name)
data.to_csv ( cvs_data_name )
print ( 'writing ' + xls_data_name)
data.to_excel ( xls_data_name, sheet_name = xls_sheet_name )
print ("done")








