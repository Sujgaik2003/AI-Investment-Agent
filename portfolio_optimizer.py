import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.plotting import plot_weights # Optional for visual debugging
import matplotlib.pyplot as plt # Needed by plot_weights if used

# Global variable to store portfolio data for caching in app.py
_portfolio_data = None

def load_portfolio_data(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Loads historical adjusted close data for multiple tickers.

    Args:
        tickers (list[str]): List of stock ticker symbols (e.g., ['SPY', 'BND', 'GLD']).
        start_date (str): The start date for data in 'YYYY-MM-DD' format.
        end_date (str): The end date for data in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A DataFrame containing the adjusted close prices for all tickers,
                      or an empty DataFrame if data cannot be fetched for all.
    """
    try:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Download data for multiple tickers
        data = yf.download(tickers, start=start, end=end)['Adj Close']
        
        if data.empty:
            print(f"No data found for portfolio tickers between {start_date} and {end_date}.")
        else:
            # Drop any rows with NaN values (e.g., if one stock has less history)
            # This is important for portfolio optimization
            data.dropna(inplace=True) 
        return data
    except Exception as e:
        print(f"Error loading portfolio data: {e}")
        return pd.DataFrame()

def optimize_portfolio(
    df_prices: pd.DataFrame, 
    risk_free_rate: float = 0.02, # Approx. current US treasury bond yield
    expected_returns_method: str = 'mean_historical_returns',
    covariance_method: str = 'ledoit_wolf'
) -> dict:
    """
    Performs portfolio optimization to find weights that maximize the Sharpe ratio.

    Args:
        df_prices (pd.DataFrame): DataFrame of historical adjusted close prices for multiple assets.
        risk_free_rate (float): The risk-free rate for Sharpe ratio calculation.
        expected_returns_method (str): Method to estimate expected returns (e.g., 'mean_historical_returns').
        covariance_method (str): Method to estimate covariance matrix (e.g., 'ledoit_wolf').

    Returns:
        dict: A dictionary containing optimal weights, performance metrics, and the EfficientFrontier object.
              Returns an empty dict if optimization fails.
    """
    if df_prices.empty or len(df_prices.columns) < 2:
        print("Insufficient data or too few assets for portfolio optimization.")
        return {}
    
    try:
        # Calculate expected returns and sample covariance matrix
        mu = expected_returns.return_model(df_prices, method=expected_returns_method)
        S = risk_models.risk_matrix(df_prices, method=covariance_method)

        # Optimize for maximal Sharpe ratio
        ef = EfficientFrontier(mu, S, weight_bounds=(0, 1)) # Weights between 0 and 1 (no shorting)
        
        # Add a small regularization to prevent extreme weights (optional, but good for stability)
        # ef.add_constraint(lambda w: w[w > 0.01].sum() >= 0.1) # Example: at least 10% in assets with >1% weight
        
        raw_weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
        cleaned_weights = ef.clean_weights() # Clean weights (e.g., round to 2 decimal places, remove tiny weights)

        # Get portfolio performance
        performance = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
        expected_annual_return, annual_volatility, sharpe_ratio = performance

        return {
            "weights": cleaned_weights,
            "expected_annual_return": expected_annual_return,
            "annual_volatility": annual_volatility,
            "sharpe_ratio": sharpe_ratio,
            "ef_object": ef # Return the EfficientFrontier object for potential further use/plotting
        }
    except Exception as e:
        print(f"Error during portfolio optimization: {e}")
        return {}

# Example usage (for testing within portfolio_optimizer.py)
if __name__ == "__main__":
    test_tickers = ['SPY', 'BND', 'GLD', 'QQQ'] # Example diversified tickers
    test_start_date = "2020-01-01"
    test_end_date = "2024-01-01"

    df_test_prices = load_portfolio_data(test_tickers, test_start_date, test_end_date)
    if not df_test_prices.empty:
        print("\nPortfolio Historical Data Head:")
        print(df_test_prices.head())

        optimization_results = optimize_portfolio(df_test_prices)
        if optimization_results:
            print("\nPortfolio Optimization Results (Max Sharpe Ratio):")
            print("Weights:", optimization_results['weights'])
            print(f"Expected Annual Return: {optimization_results['expected_annual_return']:.2%}")
            print(f"Annual Volatility: {optimization_results['annual_volatility']:.2%}")
            print(f"Sharpe Ratio: {optimization_results['sharpe_ratio']:.2f}")

            # Optional: Plotting weights (requires matplotlib)
            # if 'ef_object' in optimization_results:
            #     fig_weights = plot_weights(optimization_results['weights'])
            #     plt.show()
        else:
            print("Portfolio optimization failed for test data.")
    else:
        print("Portfolio data loading failed for test.")