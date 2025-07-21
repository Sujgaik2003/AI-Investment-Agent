import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings

# Suppress specific FutureWarning from scikit-learn related to feature names
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


def train_and_predict_rf(df: pd.DataFrame, days_ahead: int = 5) -> tuple[pd.Series, RandomForestRegressor]:
    """
    Trains a RandomForestRegressor model to predict future stock prices
    and generates predictions for a specified number of days ahead.

    Args:
        df (pd.DataFrame): DataFrame with historical stock data and technical indicators.
                           Must contain 'Close' and indicator columns (SMA_20, EMA_20, RSI, MACD).
        days_ahead (int): Number of days into the future to predict.

    Returns:
        tuple[pd.Series, RandomForestRegressor]: A tuple containing:
            - pd.Series: Predicted closing prices for future dates.
            - RandomForestRegressor: The trained model.
        Returns (pd.Series(), None) if prediction fails or data is insufficient.
    """
    if df.empty or len(df) < 50: # Need enough data for indicators and training
        print("Insufficient data for training the prediction model.")
        return pd.Series(), None

    # --- Feature Engineering ---
    # Create lagged features (previous day's close, high, low, volume)
    df_features = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df_features[f'{col}_lag1'] = df_features[col].shift(1)

    # Add technical indicators as features
    indicator_cols = ['SMA_20', 'SMA_50', 'EMA_20', 'EMA_50', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist']
    for col in indicator_cols:
        if col in df.columns:
            df_features[col] = df[col]

    # Drop rows with NaN values created by shifting/indicators
    df_features.dropna(inplace=True)

    if df_features.empty:
        print("Not enough data after feature engineering and dropping NaNs.")
        return pd.Series(), None

    # Define features (X) and target (y)
    X = df_features.drop('Close', axis=1) # Features are everything except the current 'Close'
    y = df_features['Close'] # Target is the current 'Close'

    # Align X and y by index
    common_index = X.index.intersection(y.index)
    X = X.loc[common_index]
    y = y.loc[common_index]

    if X.empty or y.empty or len(X) < 2:
        print("Insufficient data after aligning features and target for training.")
        return pd.Series(), None


    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # --- Model Training ---
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # --- Generate Future Predictions ---
    # Start with the last known *features* from your training data (X)
    last_known_features_for_prediction = X.iloc[-1].copy() # This is a pandas Series

    future_predictions = []

    # Convert to a dictionary for easier in-place modification within the loop
    current_features_dict = last_known_features_for_prediction.to_dict()

    for _ in range(days_ahead):
        # Convert the dictionary of features back to a DataFrame row for prediction
        features_df_for_pred = pd.DataFrame([current_features_dict])

        # Predict the next day's close price
        next_day_pred = model.predict(features_df_for_pred)[0]
        future_predictions.append(next_day_pred)

        # --- Update features for the *next* prediction step ---
        # The newly predicted 'Close' becomes 'Close_lag1' for the subsequent prediction.
        if 'Close_lag1' in current_features_dict:
            current_features_dict['Close_lag1'] = next_day_pred

        # For other lagged features (e.g., Open_lag1, High_lag1, Low_lag1, Volume_lag1) and indicators:
        # We simplify by keeping their values constant from the last known historical day for this POC.
        # In a real system, you'd need more complex methods to propagate or predict these auxiliary features.


    # Generate future dates (business days)
    last_historical_date = df.index[-1]
    future_dates = pd.date_range(start=last_historical_date + pd.Timedelta(days=1), periods=days_ahead, freq='B')
    
    # Ensure future_dates are unique (in case of holiday calculations leading to duplicates after initial +1 day shift)
    future_dates = future_dates.drop_duplicates()

    # Combine predictions with future dates
    predicted_series = pd.Series(future_predictions, index=future_dates, name='Predicted_Close')
    
    # In case there are more predictions than future_dates due to freq='B' and weekends
    # We should ensure the lengths match, or handle index directly
    if len(predicted_series) > len(future_dates):
        predicted_series = predicted_series.iloc[:len(future_dates)]
    predicted_series.index.name = 'Date' # Set index name for consistency


    return predicted_series, model

# Example usage (for testing within prediction_model.py)
if __name__ == "__main__":
    # Simulate some data loading and indicator addition
    from data_loader import load_stock_data, add_technical_indicators
    test_ticker = "MSFT"
    test_start_date = "2023-01-01"
    test_end_date = "2024-07-20"
    df_test = load_stock_data(test_ticker, test_start_date, test_end_date)
    if not df_test.empty:
        df_test = add_technical_indicators(df_test)
        predictions, trained_model = train_and_predict_rf(df_test, days_ahead=30)
        if not predictions.empty:
            print("\nGenerated AI Predictions (first 5):")
            print(predictions.head())
            print("\nGenerated AI Predictions (last 5):")
            print(predictions.tail())
        else:
            print("Prediction failed for test data.")
    else:
        print("Test data loading failed.")