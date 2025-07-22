from transformers import pipeline
import warnings
import requests # Ensure this is at the top level

# Suppress specific warnings from transformers/pytorch
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint were not used")
warnings.filterwarnings("ignore", message="Some weights of the model did not receive a proper `dot_product`")
warnings.filterwarnings("ignore", message="The model 'GPT2LMHeadModel' is not supported for text-generation. Please use 'CausalLMForSequenceClassification' instead.")


# Global variable to store the sentiment pipeline for caching
_sentiment_pipeline = None

def get_sentiment_pipeline():
    """
    Initializes and returns the Hugging Face sentiment analysis pipeline.
    Uses a financial-specific pre-trained model (FinBERT).
    """
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        print("Initializing FinBERT sentiment analysis pipeline... This might take a moment to download the model.")
        try:
            model_name = "ProsusAI/finbert" 
            _sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                return_all_scores=True 
            )
            print("FinBERT sentiment analysis pipeline initialized.")
        except Exception as e:
            print(f"Error initializing sentiment pipeline: {e}")
            print("Please ensure you have an active internet connection to download the model.")
            _sentiment_pipeline = None
    return _sentiment_pipeline

def analyze_sentiment(text: str) -> dict:
    """
    Analyzes the sentiment of a given text using the pre-trained NLP model (FinBERT),
    with refined neutrality handling.

    Args:
        text (str): The input text to analyze.

    Returns:
        dict: A dictionary containing 'label' (e.g., 'Positive', 'Negative', 'Neutral')
              and 'score' (confidence).
              Returns {'label': 'Error', 'score': 0.0} if pipeline is not available.
    """
    pipeline = get_sentiment_pipeline()
    if pipeline:
        try:
            raw_results = pipeline(text)[0]
            scores = {item['label']: item['score'] for item in raw_results}

            positive_score = scores.get('positive', 0.0)
            negative_score = scores.get('negative', 0.0)
            neutral_score = scores.get('neutral', 0.0)

            NEUTRAL_THRESHOLD_MIN_SCORE = 0.55
            NEUTRAL_THRESHOLD_SCORE_DIFF = 0.10
            
            max_score_label = max(scores, key=scores.get)
            max_score = scores[max_score_label]

            final_label = max_score_label
            final_score = max_score

            if neutral_score >= NEUTRAL_THRESHOLD_MIN_SCORE:
                final_label = 'neutral'
                final_score = neutral_score
            else:
                if (max_score_label != 'neutral' and (max_score - neutral_score) < NEUTRAL_THRESHOLD_SCORE_DIFF):
                    final_label = 'neutral'
                    final_score = neutral_score
                elif (positive_score < 0.4 and negative_score < 0.4 and neutral_score > 0.3):
                    final_label = 'neutral'
                    final_score = neutral_score

            label_map = {
                'negative': {'text': 'Negative', 'color': 'red'},
                'neutral': {'text': 'Neutral', 'color': 'blue'},
                'positive': {'text': 'Positive', 'color': 'green'}
            }
            mapped_label = label_map.get(final_label, {'text': 'Unknown', 'color': 'gray'})
            
            return {
                'label': mapped_label['text'],
                'color': mapped_label['color'],
                'score': final_score
            }
        except Exception as e:
            print(f"Error analyzing sentiment for text '{text[:50]}...': {e}")
            return {'label': 'Error', 'color': 'gray', 'score': 0.0}
    else:
        print("Sentiment analysis pipeline not available.")
        return {'label': 'N/A', 'color': 'gray', 'score': 0.0}


# --- Function to fetch financial news ---
def fetch_financial_news(query: str, api_key: str, page_size: int = 10) -> list[dict]:
    """
    Fetches financial news articles related to a query from NewsAPI.org.

    Args:
        query (str): The search query (e.g., stock ticker, company name).
        api_key (str): Your NewsAPI.org API key.
        page_size (int): Number of articles to fetch (max 100 for developer API).

    Returns:
        list[dict]: A list of dictionaries, each representing an article (with title, description, url).
    """
    if not api_key:
        print("NewsAPI.org API key is missing. Cannot fetch news.")
        return []

    url = "https://newsapi.org/v2/everything"
    params = {
        'q': f"{query} stock", # Search specifically for '{ticker} stock'
        'sortBy': 'relevancy', # Or 'publishedAt' for latest
        'language': 'en',
        'pageSize': page_size,
        'apiKey': api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        articles = response.json().get('articles', [])
        
        filtered_articles = [
            {"title": art.get('title', ''), "description": art.get('description', ''), "url": art.get('url', '')}
            for art in articles if art.get('title') and art.get('description') # Only include if title and description exist
        ]
        print(f"Fetched {len(filtered_articles)} news articles for '{query}'.")
        return filtered_articles
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - Check API key or usage limits.")
        if response.status_code == 401:
            print("NewsAPI: Unauthorized. Your API key might be incorrect or revoked.")
        elif response.status_code == 429:
            print("NewsAPI: Too many requests. You have hit your rate limit.")
        return []
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error occurred: {conn_err} - Check your internet connection.")
        return []
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error occurred: {timeout_err} - NewsAPI server took too long to respond.")
        return []
    except requests.exceptions.RequestException as req_err:
        print(f"An error occurred: {req_err}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred during news fetching: {e}")
        return []


# --- DEBUGGING: Confirm fetch_financial_news is defined ---
# This print statement helps confirm if Python successfully parsed the function definition.
print(f"[DEBUG: sentiment_analyzer.py] 'fetch_financial_news' is defined: {'fetch_financial_news' in globals()}")


# Example usage (for testing within sentiment_analyzer.py)
if __name__ == "__main__":
    print("Testing FinBERT sentiment analyzer with refined neutrality...")
    
    text1 = "Apple stock surged today after beating earnings estimates, showing strong growth."
    text2 = "Market is in turmoil, major tech companies are experiencing a downturn due to inflation fears."
    text3 = "The company announced steady results, no major surprises, indicating a stable period."
    text4 = "Minor fluctuations in AAPL stock, trading flat for the week."
    text5 = "Terrible news, the CEO just resigned unexpectedly."
    text6 = "The company's earnings per share (EPS) declined significantly, disappointing investors."
    text7 = "Analysts downgraded the stock to 'sell' due to poor fundamentals."
    text8 = "The firm issued a cautionary outlook for the upcoming quarter."
    text9 = "Inflation fears are subsiding, boosting market confidence."

    test_texts = [text1, text2, text3, text4, text5, text6, text7, text8, text9]

    for i, text in enumerate(test_texts):
        result = analyze_sentiment(text)
        print(f"Text {i+1}: '{text}'")
        print(f"  -> Sentiment: {result['label']} (Score: {result['score']:.2f})")

    # NOTE: You need a valid NewsAPI_KEY for the news fetching part to work here
    # This __main__ block is for independent testing of this module.
    # The actual news fetching is handled in app.py.
    # NEWS_API_KEY_TEST = "YOUR_NEWSAPI_KEY"
    # print("\nTesting news fetching (if API key is set)...")
    # if "NEWS_API_KEY_TEST" in locals() and NEWS_API_KEY_TEST != "YOUR_NEWSAPI_KEY":
    #     articles = fetch_financial_news("AAPL", NEWS_API_KEY_TEST, page_size=5)
    #     if articles:
    #         for article in articles:
    #             print(f"Title: {article['title']}")
    #             print(f"Desc: {article['description']}")
    #             sentiment = analyze_sentiment(article['description']) # Analyze description
    #             print(f"Sentiment: {sentiment['label']} (Score: {sentiment['score']:.2f})")
    #             print("---")
    #     else:
    #         print("Failed to fetch news. Check API key and internet connection.")
    # else:
    #     print("Skipping live news test: Please replace 'YOUR_NEWSAPI_KEY' with your actual key if you want to test news fetching directly in this file's __main__ block.")