import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
# from pypfopt.plotting import plot_weights # Optional for visual debugging
# import matplotlib.pyplot as plt # Needed by plot_weights if used

def load_portfolio_data(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Loads historical adjusted close data for multiple tickers.
    """
    try:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        # --- DEBUGGING: Print input parameters ---
        print(f"\n[DEBUG: load_portfolio_data] Attempting to download data for: {tickers}")
        print(f"[DEBUG: load_portfolio_data] Date Range: {start_date} to {end_date}")

        # Download data for multiple tickers
        data = yf.download(tickers, start=start, end=end)['Adj Close']
        
        # --- DEBUGGING: Print raw downloaded data info ---
        print(f"[DEBUG: load_portfolio_data] Raw data downloaded. Shape: {data.shape}")
        if data.empty:
            print("[DEBUG: load_portfolio_data] Raw data DataFrame is empty immediately after download.")
        else:
            print(f"[DEBUG: load_portfolio_data] Raw data head:\n{data.head()}")
            print(f"[DEBUG: load_portfolio_data] Raw data has NaNs (before dropna): \n{data.isnull().sum()}")


        if data.empty:
            print(f"[DEBUG: load_portfolio_data] No data found for portfolio tickers between {start_date} and {end_date}.")
        else:
            original_rows = data.shape[0]
            data.dropna(inplace=True) 
            # --- DEBUGGING: Print data info after dropna ---
            print(f"[DEBUG: load_portfolio_data] Data shape after dropna(): {data.shape}")
            print(f"[DEBUG: load_portfolio_data] Rows dropped by dropna: {original_rows - data.shape[0]}")
            if data.empty:
                print("[DEBUG: load_portfolio_data] DataFrame became EMPTY after dropping NaNs!")
            else:
                print(f"[DEBUG: load_portfolio_data] Data head after dropna():\n{data.head()}")
        return data
    except Exception as e:
        print(f"[DEBUG: load_portfolio_data] Error loading portfolio data: {e}")
        return pd.DataFrame()

def optimize_portfolio(
    df_prices: pd.DataFrame, 
    risk_free_rate: float = 0.02, 
    expected_returns_method: str = 'mean_historical_returns',
    covariance_method: str = 'ledoit_wolf'
) -> dict:
    """
    Performs portfolio optimization to find weights that maximize the Sharpe ratio.
    """
    # --- DEBUGGING: Print df_prices entering optimization ---
    print(f"\n[DEBUG: optimize_portfolio] df_prices empty? {df_prices.empty}")
    print(f"[DEBUG: optimize_portfolio] df_prices shape: {df_prices.shape}")
    if not df_prices.empty:
        print(f"[DEBUG: optimize_portfolio] df_prices head:\n{df_prices.head()}")
    
    if df_prices.empty or len(df_prices.columns) < 2:
        print("[DEBUG: optimize_portfolio] Insufficient data or too few assets for portfolio optimization.")
        return {}
    
    try:
        # Calculate expected returns and sample covariance matrix
        mu = expected_returns.return_model(df_prices, method=expected_returns_method)
        S = risk_models.risk_matrix(df_prices, method=covariance_method)
        
        # --- DEBUGGING: Print mu and S shapes ---
        print(f"[DEBUG: optimize_portfolio] Expected returns (mu) shape: {mu.shape}")
        print(f"[DEBUG: optimize_portfolio] Covariance matrix (S) shape: {S.shape}")
        
        ef = EfficientFrontier(mu, S, weight_bounds=(0, 1)) 
        raw_weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
        cleaned_weights = ef.clean_weights() 
        performance = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
        expected_annual_return, annual_volatility, sharpe_ratio = performance

        return {
            "weights": cleaned_weights,
            "expected_annual_return": expected_annual_return,
            "annual_volatility": annual_volatility,
            "sharpe_ratio": sharpe_ratio,
            "ef_object": ef
        }
    except Exception as e:
        print(f"[DEBUG: optimize_portfolio] Error during portfolio optimization: {e}")
        return {}

# Example usage (for testing within portfolio_optimizer.py) - no changes here