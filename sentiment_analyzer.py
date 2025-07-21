from transformers import pipeline
import warnings
import requests # NEW: Import requests library

# Suppress specific warnings from transformers/pytorch
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint were not used")
warnings.filterwarnings("ignore", message="Some weights of the model did not receive a proper `dot_product`")

# Global variable to store the sentiment pipeline for caching
_sentiment_pipeline = None

def get_sentiment_pipeline():
    """
    Initializes and returns the Hugging Face sentiment analysis pipeline.
    Uses a common pre-trained model for general sentiment.
    """
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        print("Initializing sentiment analysis pipeline... This might take a moment to download the model.")
        try:
            _sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            print("Sentiment analysis pipeline initialized.")
        except Exception as e:
            print(f"Error initializing sentiment pipeline: {e}")
            print("Please ensure you have an active internet connection to download the model.")
            _sentiment_pipeline = None # Reset to None if initialization fails
    return _sentiment_pipeline

def analyze_sentiment(text: str) -> dict:
    """
    Analyzes the sentiment of a given text using the pre-trained NLP model.

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
            result = pipeline(text)[0]
            
            # Map labels to more readable strings and colors
            label_map = {
                'LABEL_0': {'text': 'Negative', 'color': 'red'},
                'LABEL_1': {'text': 'Neutral', 'color': 'blue'},
                'LABEL_2': {'text': 'Positive', 'color': 'green'}
            }
            mapped_label = label_map.get(result['label'], {'text': 'Unknown', 'color': 'gray'})
            
            return {
                'label': mapped_label['text'],
                'color': mapped_label['color'],
                'score': result['score']
            }
        except Exception as e:
            print(f"Error analyzing sentiment for text '{text[:50]}...': {e}")
            return {'label': 'Error', 'color': 'gray', 'score': 0.0}
    else:
        print("Sentiment analysis pipeline not available.")
        return {'label': 'N/A', 'color': 'gray', 'score': 0.0}


# --- NEW: Function to fetch financial news ---
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

    # Using 'everything' endpoint for broader search, could also use 'top-headlines' with category=business
    url = "https://newsapi.org/v2/everything"
    params = {
        'q': f"{query} stock", # Search specifically for '{ticker} stock'
        'sortBy': 'relevancy', # Or 'publishedAt' for latest
        'language': 'en',
        'pageSize': page_size,
        'apiKey': api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=10) # 10-second timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        articles = response.json().get('articles', [])
        
        # Filter out articles with missing title or description
        filtered_articles = [
            {"title": art.get('title', ''), "description": art.get('description', ''), "url": art.get('url', '')}
            for art in articles if art.get('title') and art.get('description')
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


# Example usage (for testing within sentiment_analyzer.py)
if __name__ == "__main__":
    print("Testing sentiment analyzer...")
    # NOTE: You need a valid NewsAPI_KEY for the news fetching part to work here
    NEWS_API_KEY_TEST = "8e08a146509d477da66326569c670b4c" # Replace with your actual key for testing
    
    # Test sentiment analysis
    text1 = "Apple stock surged today after beating earnings estimates, showing strong growth."
    text2 = "Market is in turmoil, major tech companies are experiencing a downturn due to inflation fears."
    print(f"'{text1}' -> {analyze_sentiment(text1)}")
    print(f"'{text2}' -> {analyze_sentiment(text2)}")

    print("\nTesting news fetching...")
    if NEWS_API_KEY_TEST != "YOUR_NEWSAPI_KEY":
        articles = fetch_financial_news("AAPL", NEWS_API_KEY_TEST, page_size=5)
        if articles:
            for article in articles:
                print(f"Title: {article['title']}")
                print(f"Desc: {article['description']}")
                print(f"URL: {article['url']}")
                sentiment = analyze_sentiment(article['description']) # Analyze description
                print(f"Sentiment: {sentiment['label']} (Score: {sentiment['score']:.2f})")
                print("---")
        else:
            print("Failed to fetch news. Check API key and internet connection.")
    else:
        print("Skipping live news test: Please replace 'YOUR_NEWSAPI_KEY' with your actual key in sentiment_analyzer.py's __main__ block.")