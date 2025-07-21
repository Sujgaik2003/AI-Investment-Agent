import yfinance as yf
import pandas as pd
from datetime import datetime
import ta # Import the technical analysis library

def load_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Loads historical stock data for a given ticker from Yahoo Finance.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL', 'MSFT').
        start_date (str): The start date for data in 'YYYY-MM-DD' format.
        end_date (str): The end date for data in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A DataFrame containing the historical stock data,
                      or an empty DataFrame if data cannot be fetched.
    """
    try:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            print(f"No data found for {ticker} between {start_date} and {end_date}. Check ticker or date range.")
        return data
    except Exception as e:
        print(f"Error loading data for {ticker}: {e}")
        return pd.DataFrame() # Return empty DataFrame on error

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds various technical indicators to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with 'High', 'Low', 'Close', 'Volume' columns.

    Returns:
        pd.DataFrame: DataFrame with added technical indicator columns.
    """
    if df.empty:
        return df

    # Ensure 'Close', 'High', 'Low', 'Volume' columns are 1-dimensional Series of float type
    # This prevents TypeErrors from the ta library
    close_prices = df['Close'].astype(float).squeeze()
    high_prices = df['High'].astype(float).squeeze()
    low_prices = df['Low'].astype(float).squeeze()
    volume_data = df['Volume'].astype(float).squeeze()

    # Simple Moving Average (SMA)
    df['SMA_20'] = ta.trend.sma_indicator(close_prices, window=20)
    df['SMA_50'] = ta.trend.sma_indicator(close_prices, window=50)

    # Exponential Moving Average (EMA)
    df['EMA_20'] = ta.trend.ema_indicator(close_prices, window=20)
    df['EMA_50'] = ta.trend.ema_indicator(close_prices, window=50)

    # Relative Strength Index (RSI)
    df['RSI'] = ta.momentum.rsi(close_prices, window=14)

    # Moving Average Convergence Divergence (MACD)
    df['MACD'] = ta.trend.macd(close_prices)
    df['MACD_Signal'] = ta.trend.macd_signal(close_prices)
    df['MACD_Hist'] = ta.trend.macd_diff(close_prices)

    return df

# Example of how to use this function (for testing within data_loader.py)
if __name__ == "__main__":
    test_ticker = "GOOG"
    test_start_date = "2023-01-01"
    test_end_date = "2024-01-01"
    stock_df = load_stock_data(test_ticker, test_start_date, test_end_date)
    if not stock_df.empty:
        stock_df_with_indicators = add_technical_indicators(stock_df)
        print(f"Successfully loaded and added indicators for {test_ticker}:")
        print(stock_df_with_indicators.tail()) # Show last few rows with new indicator columns