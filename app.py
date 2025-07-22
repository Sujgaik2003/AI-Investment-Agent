import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Import our custom modules
from data_loader import load_stock_data, add_technical_indicators
from prediction_model import train_and_predict_rf
from sentiment_analyzer import analyze_sentiment, fetch_financial_news
from portfolio_optimizer import load_portfolio_data, optimize_portfolio
from email_generator import generate_investment_email # NEW: Import email generation function

st.set_page_config(layout="wide", page_title="AI Investment Agent POC")

st.title("📈 AI Investment Agent: Stock Price & Portfolio Analyzer (POC)")
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

st.sidebar.markdown("---")
st.sidebar.subheader("API Keys (for live features)")
news_api_key = st.sidebar.text_input("NewsAPI.org API Key", type="password")

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

# --- Caching News Fetching ---
@st.cache_data(ttl=600) # Cache news for 10 minutes (600 seconds)
def get_and_analyze_news_sentiment(ticker, api_key):
    """
    Fetches news and analyzes sentiment, caching the results.
    """
    if not api_key:
        return {}, [], "NewsAPI.org API Key is required for live sentiment analysis."
    
    articles = fetch_financial_news(ticker, api_key, page_size=20) # Fetch more articles
    
    if not articles:
        return {}, [], "Could not fetch news. Check API key, internet, or ticker relevance."

    sentiment_results = []
    for article in articles:
        sentiment = analyze_sentiment(article['description'] if article['description'] else article['title'])
        sentiment_results.append({
            'title': article['title'],
            'description': article['description'],
            'url': article['url'],
            'label': sentiment['label'],
            'color': sentiment['color'],
            'score': sentiment['score']
        })

    # Aggregate overall sentiment
    overall_scores = [res['score'] for res in sentiment_results if res['label'] != 'N/A' and res['label'] != 'Error']
    
    if not overall_scores:
        return {}, [], "No valid sentiment scores could be generated from fetched news."

    average_score = np.mean(overall_scores)

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

    overall_sentiment_info = {
        'label': overall_sentiment_label,
        'color': overall_sentiment_color,
        'average_score': average_score
    }
    return overall_sentiment_info, sentiment_results, ""

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


# --- Main app logic ---
if st.sidebar.button("Load Data"):
    st.subheader(f"Analyzing {ticker_symbol}")

    data = get_and_process_stock_data(
        ticker_symbol,
        start_date_input.strftime('%Y-%m-%d'),
        end_date_input.strftime('%Y-%m-%d')
    )

    # Variables to pass to email generator (initialize as None/empty)
    predictions_summary_text = "Not available."
    portfolio_advice_summary_text = "Not available."
    overall_sentiment_info = {'label': 'N/A', 'color': 'gray', 'average_score': 0.0}

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
                predictions_summary_text = f"The AI model predicts {ticker_symbol} to have a {predictions.iloc[-1]:.2f} price in 30 days (from {predictions.iloc[0]:.2f} today)." # Summarize prediction  
            else:
                st.warning("Could not generate AI Price Prediction. Check data sufficiency for the model (e.g., ensure enough historical data).")


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

        # --- Actual NLP Market Sentiment Display ---
        st.markdown("---")
        st.subheader("Market Sentiment Analysis (Powered by NLP)")
        
        overall_sentiment_info, sentiment_results, error_message = get_and_analyze_news_sentiment(ticker_symbol, news_api_key)

        if error_message:
            st.error(error_message)
        elif overall_sentiment_info:
            st.markdown(f"**Overall Market Sentiment for {ticker_symbol}:** <span style='color:{overall_sentiment_info['color']}; font-size: 24px;'>**{overall_sentiment_info['label']}**</span>", unsafe_allow_html=True)
            st.write(f"*(Based on analysis of {len(sentiment_results)} recent news articles - Average Score: {overall_sentiment_info['average_score']:.2f})*")
            st.write("""
            *Note: This sentiment is now powered by a **pre-trained NLP model** from Hugging Face Transformers,
            analyzing real-time financial news from NewsAPI.org.*
            """)

            with st.expander("Show Recent News & Individual Sentiment Details"):
                if sentiment_results:
                    for article_info in sentiment_results:
                        st.markdown(f"**Title:** [{article_info['title']}]({article_info['url']})")
                        st.markdown(f"**Description:** {article_info['description']}")
                        st.markdown(f"**Sentiment:** <span style='color:{article_info['color']};'>**{article_info['label']}**</span> (Score: {article_info['score']:.2f})", unsafe_allow_html=True)
                        st.markdown("---")
                else:
                    st.write("No news articles fetched or processed for sentiment.")
        else:
            st.info("No sentiment analysis results available.")

        # --- Investment Plan & Allocation Advice (Placeholder) ---
        st.markdown("---")
        st.subheader("Investment Plan & Allocation Advice")

        mock_risk_tolerance = st.select_slider(
            "Select your Mock Risk Tolerance (for placeholder advice):",
            options=['Low', 'Medium', 'High'],
            value='Medium'
        )

        # --- AI-Driven Portfolio Optimization ---
        st.markdown("---")
        st.subheader("AI-Driven Portfolio Optimization (Max Sharpe Ratio)")

        # Define a fixed set of diversified tickers for portfolio optimization POC
        portfolio_tickers = ['SPY', 'BND', 'GLD', 'QQQ', 'VOO'] # S&P 500, Bonds, Gold, Nasdaq 100, S&P 500
        
        # Determine appropriate date range for portfolio data. Use a longer range than 1 year for better optimization
        portfolio_end_date = datetime.now()
        portfolio_start_date = portfolio_end_date - timedelta(days=5 * 365) # 5 years of data

        @st.cache_data(ttl=3600) # Cache portfolio data as well
        def get_and_optimize_portfolio(tickers, start, end):
            df_prices = load_portfolio_data(tickers, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
            if not df_prices.empty:
                results = optimize_portfolio(df_prices)
                return results
            return {}

        optimization_results = get_and_optimize_portfolio(portfolio_tickers, portfolio_start_date, portfolio_end_date)

        if optimization_results:
            st.success("Portfolio Optimized for Maximum Sharpe Ratio!")
            st.write("---")
            st.markdown(f"**Expected Annual Return:** `{optimization_results['expected_annual_return']:.2%}`")
            st.markdown(f"**Annual Volatility:** `{optimization_results['annual_volatility']:.2%}`")
            st.markdown(f"**Sharpe Ratio:** `{optimization_results['sharpe_ratio']:.2f}`")
            st.write("---")
            st.write("### Recommended Asset Allocation:")
            
            # Convert weights to DataFrame for better display
            weights_df = pd.DataFrame(list(optimization_results['weights'].items()), columns=['Asset', 'Weight'])
            weights_df['Weight'] = weights_df['Weight'].apply(lambda x: f"{x:.2%}") # Format as percentage
            st.dataframe(weights_df, use_container_width=True)
            
            portfolio_advice_summary_text = f"An optimized portfolio suggests allocating assets as follows: {', '.join([f'{asset}: {weight}' for asset, weight in optimization_results['weights'].items() if float(weight.replace('%', '').strip()) > 0])}. Expected annual return is {optimization_results['expected_annual_return']:.2%} with {optimization_results['annual_volatility']:.2%} volatility."

            st.write("""
            *Note: This portfolio optimization is based on historical data for a fixed set of assets and aims to maximize the Sharpe Ratio.
            In a full AI system, this would be highly customized based on your risk profile, financial goals,
            and real-time market predictions, potentially using more advanced optimization techniques (e.g., Reinforcement Learning, Goal-Based Investing).*
            """)
        else:
            st.warning("Could not perform portfolio optimization. Check data availability or asset list (e.g., for SPY, BND, GLD, QQQ, VOO).")

        # Re-add the previous risk tolerance advice, as it's separate from optimization
        if mock_risk_tolerance == 'Low':
            st.markdown("---") # Separator for clarity
            st.info("Your **Low Risk** profile suggests a more conservative allocation. Consider the optimized portfolio above or a traditional split like:")
            st.markdown("""
            * **60% Bonds/Fixed Income:** Focus on stable, income-generating assets.
            * **30% Diversified ETFs:** Broad market exposure, less volatility.
            * **10% Cash/Money Market:** For liquidity and stability.
            """)
        elif mock_risk_tolerance == 'Medium':
            st.markdown("---")
            st.info("Your **Medium Risk** profile balances growth and stability. Consider the optimized portfolio or a traditional split like:")
            st.markdown("""
            * **40% Large-Cap Stocks:** Established companies with stable growth.
            * **30% Bonds/Fixed Income:** For diversification and risk reduction.
            * **20% Mid-Cap/Small-Cap Stocks:** Potential for higher growth, but more volatility.
            * **10% Real Estate/Alternatives:** For further diversification.
            """)
        else: # High Risk
            st.markdown("---")
            st.info("Your **High Risk** profile seeks growth. Consider the optimized portfolio or a traditional split like:")
            st.markdown("""
            * **60% Growth Stocks (Tech, Biotech):** High growth potential, higher volatility.
            * **20% Emerging Markets ETFs:** Exposure to high-growth economies.
            * **10% Cryptocurrencies/Venture Capital:** Speculative, high-reward potential.
            * **10% Cash/Short-Term Bonds:** For tactical opportunities and some stability.
            """)
        
        # --- NEW FEATURE: AI-Generated Investment Summary Email ---
        st.markdown("---")
        st.subheader("AI-Generated Investment Summary Email")
        
        if st.button("Generate Email Draft"):
            email_draft = generate_investment_email(
                ticker_symbol,
                overall_sentiment_info,
                predictions_summary_text,
                portfolio_advice_summary_text
            )
            st.text_area("Investment Summary Email Draft", email_draft, height=400)
            st.write("""
            *Note: This email is generated by a **basic LLM (GPT-2)** and is for demonstration purposes only.
            For production, you would use more advanced LLMs fine-tuned for financial communication and integrate with email services.*
            """)
        # --- END NEW FEATURE ---


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
    * **Actual Market Sentiment Analysis (Basic NLP):** Integrates a pre-trained NLP model to analyze sentiment from **real-time news headlines** (requires NewsAPI.org key).
    * **AI-Driven Portfolio Optimization (Basic):** Calculates optimal asset weights for a sample portfolio using Modern Portfolio Theory.
    * **Mock Investment Advice:** Offers sample portfolio allocation advice based on a selected risk tolerance, demonstrating future personalization.
    * **AI-Generated Email Summary:** Uses a Large Language Model to draft an email summarizing the analysis.

    **Technologies Used:**
    * **Python:** The core programming language.
    * **Streamlit:** For rapid development of the interactive web application and UI.
    * **Plotly:** For creating rich, interactive data visualizations.
    * **Pandas:** For powerful data manipulation and analysis.
    * **yfinance:** For fetching historical stock data.
    * **ta:** For calculating technical analysis indicators.
    * **scikit-learn:** For machine learning model implementation.
    * **NumPy:** For numerical operations.
    * **Hugging Face Transformers / PyTorch:** For pre-trained NLP sentiment analysis and text generation (LLM).
    * **Requests:** For making HTTP calls to APIs (e.g., NewsAPI.org).
    * **PyPortfolioOpt / cvxpy:** For portfolio optimization.

    ---

    **❗ DISCLAIMER ❗**
    **This application is for demonstration and educational purposes only. It provides simulated data, basic AI predictions, and placeholder advice. It is NOT intended to be, and should not be construed as, financial advice, investment recommendations, or an endorsement of any particular investment or strategy.**

    **Investing in financial markets involves risks, including the potential loss of principal. Past performance is not indicative of future results.**

    **Always consult with a qualified financial professional before making any investment decisions. Do not rely solely on automated tools or models for your investment choices.**
    """)