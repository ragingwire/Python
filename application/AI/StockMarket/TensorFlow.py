import yfinance as yf
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Step 1: Download historical stock data for TXN from Yahoo Finance
def download_stock_data(ticker, start_date, end_date):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if stock_data.empty:
            raise ValueError("No data retrieved from Yahoo Finance.")
        print("Downloaded data sample:")
        print(stock_data.head())
        return stock_data
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

# Step 2: Prepare data for Prophet
def prepare_prophet_data(stock_data):
    try:
        # Select 'Date' and 'Close' columns, rename to 'ds' and 'y'
        df = stock_data[['Close']].reset_index()
        df = df.rename(columns={'Date': 'ds', 'Close': 'y'})
        
        # Ensure 'ds' is datetime
        df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
        
        # Ensure 'y' is numeric (float)
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        
        # Drop rows with missing or invalid values
        initial_rows = len(df)
        df = df.dropna(subset=['ds', 'y'])
        if len(df) < initial_rows:
            print(f"Dropped {initial_rows - len(df)} rows due to missing or invalid values.")
        
        # Check if DataFrame is empty
        if df.empty:
            raise ValueError("DataFrame is empty after cleaning.")
        
        # Verify data types
        print("Prepared data sample:")
        print(df.head())
        print("Data types:", df.dtypes)
        
        return df
    except Exception as e:
        print(f"Error preparing data: {e}")
        return None

# Step 3: Train Prophet model and make predictions
def train_and_predict(df, forecast_periods=365):
    try:
        # Initialize Prophet model
        model = Prophet(daily_seasonality=True)
        
        # Debug: Check input data before fitting
        print("Input data shape:", df.shape)
        print("Input data columns:", df.columns)
        
        # Fit the model
        model.fit(df)
        
        # Create future dataframe for predictions
        future = model.make_future_dataframe(periods=forecast_periods)
        
        # Generate predictions
        forecast = model.predict(future)
        print("Forecast sample:")
        print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
        return model, forecast
    except Exception as e:
        print(f"Error in model training or prediction: {e}")
        return None, None

# Step 4: Visualize the results
def plot_forecast(model, forecast, ticker):
    try:
        # Plot the forecast
        fig = model.plot(forecast)
        plt.title(f'{ticker} Stock Price Forecast')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        
        # Add legend
        plt.legend(['Actual', 'Forecast', 'Uncertainty Interval'], loc='upper left')
        
        # Adjust layout
        plt.tight_layout()
        
        # Show plot
        plt.show()
    except Exception as e:
        print(f"Error in plotting: {e}")

# Main execution
def main():
    ticker = "TXN"
    
    # Define date range (last 5 years of historical data)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    
    # Download data
    stock_data = download_stock_data(ticker, start_date, end_date)
    if stock_data is None:
        print("Failed to download stock data. Exiting.")
        return
    
    # Prepare data for Prophet
    prophet_df = prepare_prophet_data(stock_data)
    if prophet_df is None:
        print("Failed to prepare data for Prophet. Exiting.")
        return
    
    # Train model and predict
    model, forecast = train_and_predict(prophet_df)
    if model is None or forecast is None:
        print("Failed to train model or generate predictions. Exiting.")
        return
    
    # Plot results
    plot_forecast(model, forecast, ticker)

if __name__ == "__main__":
    main()