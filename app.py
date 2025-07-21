import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from data_loader import load_stock_data, add_technical_indicators
from prediction_model import train_and_predict_rf
from sentiment_analyzer import analyze_sentiment # NEW Import for sentiment analysis

st.set_page_config(layout="wide", page_title="AI Investment Agent POC")

st.title("📈 AI Investment Agent: Stock Price & Volume Visualizer (POC)")
st.markdown("---")

# --- Sidebar for user input ---
st.sidebar.header("Input Parameters")
ticker_symbol = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, GOOG)", "AAPL").upper()

# Default date range: last 1 year
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

# Date input widgets
start_date_input = st.sidebar.date_input("Start Date", start_date)
end_date_input = st.sidebar.date_input("End Date", end_date)

# Ensure start date is before end date
if start_date_input > end_date_input:
    st.sidebar.error("Error: Start date cannot be after end date.")
    st.stop()

# Technical Indicator Checkboxes
st.sidebar.subheader("Technical Indicators")
show_sma = st.sidebar.checkbox("Show SMA (20, 50)", True)
show_ema = st.sidebar.checkbox("Show EMA (20, 50)", False)
show_rsi = st.sidebar.checkbox("Show RSI", False)
show_macd = st.sidebar.checkbox("Show MACD", False)
show_volume = st.sidebar.checkbox("Show Volume", True)
st.sidebar.markdown("---")
show_ai_prediction = st.sidebar.checkbox("Show AI Price Prediction", True)

# --- Caching Functions ---
@st.cache_data(ttl=3600) # Cache data for 1 hour (3600 seconds)
def get_and_process_stock_data(ticker, start_date, end_date):
    """
    Fetches and processes stock data with technical indicators, and caches the result.
    """
    data = load_stock_data(ticker, start_date, end_date)
    if not data.empty:
        data = add_technical_indicators(data)
    return data

# --- Helper function for mock prediction ---
def generate_mock_prediction(last_date: datetime, last_price: float, days_ahead: int = 30) -> pd.DataFrame:
    """
    Generates mock future dates and prices for a placeholder prediction.
    For simplicity, predicts a slight upward trend.
    """
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_ahead + 1)]
    future_prices = [last_price * (1 + (i * 0.001) + (i % 5 * 0.0005)) for i in range(1, days_ahead + 1)]

    mock_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Close': future_prices
    }).set_index('Date')
    return mock_df

# --- Dynamic Mock Sentiment Generator (OLD - to be replaced or adapted) ---
# This function is now superseded by the actual NLP sentiment analysis.
# We will remove the definition here and use the analyze_sentiment from the new module.
# The code below will directly use analyze_sentiment() in the main logic.

# --- Main app logic ---
if st.sidebar.button("Load Data"):
    st.subheader(f"Analyzing {ticker_symbol}")

    data = get_and_process_stock_data(
        ticker_symbol,
        start_date_input.strftime('%Y-%m-%d'),
        end_date_input.strftime('%Y-%m-%d')
    )

    if not data.empty:
        # --- Create Subplots for Price and Volume ---
        rows = 1
        row_heights = [0.7] # Main chart
        if show_macd:
            rows += 1
            row_heights.append(0.3)
        if show_rsi:
            rows += 1
            row_heights.append(0.3)
        if show_volume:
            rows += 1
            row_heights.append(0.2)

        subplot_titles = [f'Price Chart for {ticker_symbol}']
        if show_macd: subplot_titles.append('Moving Average Convergence Divergence (MACD)')
        if show_rsi: subplot_titles.append('Relative Strength Index (RSI)')
        if show_volume: subplot_titles.append('Volume')


        fig = make_subplots(
            rows=rows,
            cols=1,
            vertical_spacing=0.02,
            row_heights=row_heights,
            subplot_titles=subplot_titles
        )

        # --- Add Candlestick trace to the first subplot ---
        fig.add_trace(go.Candlestick(x=data.index,
                                    open=data['Open'],
                                    high=data['High'],
                                    low=data['Low'],
                                    close=data['Close'],
                                    name='Candlestick'),
                      row=1, col=1)

        # --- Add Technical Indicator Traces to the first subplot ---
        if show_sma:
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], mode='lines', name='SMA 20', line=dict(color='blue', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], mode='lines', name='SMA 50', line=dict(color='orange', width=1)), row=1, col=1)
        if show_ema:
            fig.add_trace(go.Scatter(x=data.index, y=data['EMA_20'], mode='lines', name='EMA 20', line=dict(color='purple', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['EMA_50'], mode='lines', name='EMA 50', line=dict(color='brown', width=1)), row=1, col=1)

        # --- Add AI Price Prediction Trace ---
        if show_ai_prediction and not data.empty:
            predictions, trained_model = train_and_predict_rf(data.copy(), days_ahead=30)

            if not predictions.empty:
                combined_x = sorted(pd.concat([pd.Series(data.index), pd.Series(predictions.index)]).unique())
                combined_y = pd.Series(index=combined_x, dtype=float)
                combined_y.update(data['Close'].squeeze().astype(float))
                combined_y.update(predictions.squeeze().astype(float))

                fig.add_trace(go.Scatter(x=combined_x, y=combined_y, mode='lines', name='AI Price Prediction',
                                         line=dict(color='red', width=2, dash='dash'),
                                         hovertemplate='Date: %{x}<br>Predicted Price: %{y:.2f}<extra></extra>'),
                              row=1, col=1)
                st.info("AI Price Prediction generated using a Random Forest Regressor. This is a basic model for POC.")
            else:
                st.warning("Could not generate AI Price Prediction. Check data sufficiency.")


        # --- Add MACD, RSI, Volume traces to their own subplots based on dynamic row count ---
        current_plot_row = 1 # Tracks the current row for plotting

        if show_macd:
            current_plot_row += 1
            fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], mode='lines', name='MACD Line', line=dict(color='blue')), row=current_plot_row, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], mode='lines', name='Signal Line', line=dict(color='red')), row=current_plot_row, col=1)
            fig.add_trace(go.Bar(x=data.index, y=data['MACD_Hist'], name='Histogram', marker_color='grey'), row=current_plot_row, col=1)
            fig.update_yaxes(title_text="MACD", row=current_plot_row, col=1)

        if show_rsi:
            current_plot_row += 1
            fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI', line=dict(color='green')), row=current_plot_row, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_plot_row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_plot_row, col=1)
            fig.update_yaxes(title_text="RSI", row=current_plot_row, col=1, range=[0, 100])

        if show_volume:
            current_plot_row += 1
            fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='grey'), row=current_plot_row, col=1)
            fig.update_yaxes(title_text="Volume", row=current_plot_row, col=1)

        # --- Manual X-axis Linking and Label Hiding ---
        for i in range(1, rows):
            fig.update_xaxes(matches='x', showticklabels=False, row=i, col=1)

        # --- Final Layout Updates ---
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            height=900,
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

        st.success("Data loaded and charts displayed successfully!")

        # --- NEW FEATURE: Actual NLP Market Sentiment Display ---
        st.markdown("---")
        st.subheader("Market Sentiment Analysis (Powered by NLP)")
        
        # Sample texts for demonstration (in a real app, these would come from a news API)
        sample_news_texts = [
            f"{ticker_symbol} shares soared today after positive earnings report.",
            f"Analysts are cautious about {ticker_symbol} outlook due to macro economic headwinds.",
            f"The market reacted negatively to the latest inflation data, causing a dip in {ticker_symbol} stock.",
            f"{ticker_symbol} announces innovative new product line, surprising competitors.",
            f"Minor fluctuations in {ticker_symbol} stock, trading flat for the week."
        ]

        # Use an expander to show individual sentiment scores for sample texts
        with st.expander("Show Sample News Sentiment Details"):
            for i, text in enumerate(sample_news_texts):
                sentiment_result = analyze_sentiment(text)
                st.markdown(f"**News Snippet {i+1}:** \"{text}\"")
                st.markdown(f"**Sentiment:** <span style='color:{sentiment_result['color']};'>**{sentiment_result['label']}**</span> (Score: {sentiment_result['score']:.2f})", unsafe_allow_html=True)
                st.markdown("---")
        
        # You could also try to derive an 'overall' sentiment based on these.
        # For a simple overall sentiment, let's just pick one of the sample texts, or average scores.
        # For this POC, let's take a simple average of sentiment scores to determine an overall sentiment.
        # This is a simplification; a real system would aggregate from many sources.

        overall_scores = [analyze_sentiment(text)['score'] for text in sample_news_texts]
        average_score = np.mean(overall_scores)

        # Map average score to a general sentiment
        if average_score > 0.7:
            overall_sentiment_label = "Strongly Positive"
            overall_sentiment_color = "darkgreen"
        elif average_score > 0.55:
            overall_sentiment_label = "Positive"
            overall_sentiment_color = "green"
        elif average_score < 0.3:
            overall_sentiment_label = "Strongly Negative"
            overall_sentiment_color = "darkred"
        elif average_score < 0.45:
            overall_sentiment_label = "Negative"
            overall_sentiment_color = "red"
        else:
            overall_sentiment_label = "Neutral/Mixed"
            overall_sentiment_color = "blue"

        st.markdown(f"**Overall Market Sentiment for {ticker_symbol}:** <span style='color:{overall_sentiment_color}; font-size: 24px;'>**{overall_sentiment_label}**</span>", unsafe_allow_html=True)
        st.write(f"*(Based on analysis of sample news snippets - Average Score: {average_score:.2f})*")
        st.write("""
        *Note: This sentiment is now powered by a **pre-trained NLP model** from Hugging Face Transformers.
        In a full AI system, this would analyze real-time financial news feeds, social media data,
        and corporate reports specific to the selected ticker.*
        """)
        # --- END NEW FEATURE ---

        # --- Investment Plan & Allocation Advice (Placeholder) ---
        st.markdown("---")
        st.subheader("Investment Plan & Allocation Advice")

        mock_risk_tolerance = st.select_slider(
            "Select your Mock Risk Tolerance (for placeholder advice):",
            options=['Low', 'Medium', 'High'],
            value='Medium'
        )

        if mock_risk_tolerance == 'Low':
            st.success("For a **Low Risk** profile, consider a conservative portfolio:")
            st.markdown("""
            * **60% Bonds/Fixed Income:** Focus on stable, income-generating assets.
            * **30% Diversified ETFs:** Broad market exposure, less volatility.
            * **10% Cash/Money Market:** For liquidity and stability.
            """)
        elif mock_risk_tolerance == 'Medium':
            st.success("For a **Medium Risk** profile, a balanced approach is often suitable:")
            st.markdown("""
            * **40% Large-Cap Stocks:** Established companies with stable growth.
            * **30% Bonds/Fixed Income:** For diversification and risk reduction.
            * **20% Mid-Cap/Small-Cap Stocks:** Potential for higher growth, but more volatility.
            * **10% Real Estate/Alternatives:** For further diversification.
            """)
        else: # High Risk
            st.success("For a **High Risk** profile, focus on growth opportunities:")
            st.markdown("""
            * **60% Growth Stocks (Tech, Biotech):** High growth potential, higher volatility.
            * **20% Emerging Markets ETFs:** Exposure to high-growth economies.
            * **10% Cryptocurrencies/Venture Capital:** Speculative, high-reward potential.
            * **10% Cash/Short-Term Bonds:** For tactical opportunities and some stability.
            """)

        st.write("""
        *Note: This advice is a **placeholder** based on a simple mock risk tolerance.
        In a full AI system, this would be generated by sophisticated portfolio
        optimization models considering your detailed financial goals,
        risk capacity, predicted market movements, and personalized
        behavioral insights.*
        """)


    else:
        st.warning(f"Could not load data for {ticker_symbol}. Please check the ticker symbol and date range.")

# --- Global Disclaimer and About Section ---
st.markdown("---")
with st.expander("About this AI Investment Agent (POC) & Disclaimer"):
    st.markdown("""
    This application is a **Proof of Concept (POC)** for an AI-driven investment agent, built entirely in Python using Streamlit for the user interface.

    **Key Features Demonstrated:**
    * **Interactive Stock Charting:** Displays historical stock data (Candlestick, Volume) using `yfinance` and `Plotly`.
    * **Technical Indicators:** Calculates and visualizes common technical indicators (SMA, EMA, RSI, MACD) using the `ta` library.
    * **Data Caching:** Leverages Streamlit's caching mechanisms for improved performance.
    * **AI Price Prediction (Basic):** Integrates a basic Random Forest Regressor to forecast future stock prices.
    * **Actual Market Sentiment Analysis (Basic NLP):** Integrates a pre-trained NLP model to analyze sentiment from sample news snippets.
    * **Mock Investment Advice:** Offers sample portfolio allocation advice based on a selected risk tolerance, demonstrating future personalization.

    **Technologies Used:**
    * **Python:** The core programming language.
    * **Streamlit:** For rapid development of the interactive web application and UI.
    * **Plotly:** For creating rich, interactive data visualizations.
    * **Pandas:** For powerful data manipulation and analysis.
    * **yfinance:** For fetching historical stock data.
    * **ta:** For calculating technical analysis indicators.
    * **scikit-learn:** For machine learning model implementation.
    * **NumPy:** For numerical operations.
    * **Hugging Face Transformers / PyTorch:** For pre-trained NLP sentiment analysis.

    ---

    **❗ DISCLAIMER ❗**
    **This application is for demonstration and educational purposes only. It provides simulated data, basic AI predictions, and placeholder advice. It is NOT intended to be, and should not be construed as, financial advice, investment recommendations, or an endorsement of any particular investment or strategy.**

    **Investing in financial markets involves risks, including the potential loss of principal. Past performance is not indicative of future results.**

    **Always consult with a qualified financial professional before making any investment decisions. Do not rely solely on automated tools or models for your investment choices.**
    """)