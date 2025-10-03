# Note: Install required libraries before running:
# pip install yfinance prophet matplotlib pandas

import yfinance as yf
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Download historical stock data for TXN from Yahoo Finance
ticker = 'TXN'
data = yf.download(ticker, period='max', progress=False)


if data.empty:
    print(f"No data found for ticker {ticker}. Please check the ticker symbol.")
else:
    # Prepare data for Prophet: use 'Close' price
    df = data[['Close']].reset_index()
    df.columns = ['ds', 'y']  # Prophet requires 'ds' for date, 'y' for value
    df['ds'] = pd.to_datetime(df['ds'])  # Ensure date is datetime

    # Initialize and fit the Prophet model
    model = Prophet(daily_seasonality=True)  # Enable daily seasonality for stock data
    model.fit(df)

    # Create future dataframe for forecasting (next 365 days)
    future = model.make_future_dataframe(periods=365)

    # Generate forecast
    forecast = model.predict(future)

    # Visualize the forecast
    fig = model.plot(forecast)
    plt.title(f'{ticker} Stock Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.show()

    # Plot components (trend, seasonality)
    fig2 = model.plot_components(forecast)
    plt.show()