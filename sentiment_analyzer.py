from transformers import pipeline
import warnings

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
        # Using a general sentiment model (e.g., for positive/negative/neutral)
        # 'distilbert-base-uncased-finetuned-sst-2-english' is good for positive/negative
        # For a more nuanced 'neutral', a different model or custom logic might be needed.
        # Let's use 'cardiffnlp/twitter-roberta-base-sentiment-latest' which explicitly handles NEUTRAL
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
            # The model returns a list of dictionaries, e.g., [{'label': 'LABEL_2', 'score': 0.999}]
            # LABEL_0: Negative, LABEL_1: Neutral, LABEL_2: Positive for this model
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

# Example usage (for testing within sentiment_analyzer.py)
if __name__ == "__main__":
    print("Testing sentiment analyzer...")
    text1 = "Apple stock surged today after beating earnings estimates, showing strong growth."
    text2 = "Market is in turmoil, major tech companies are experiencing a downturn due to inflation fears."
    text3 = "The company announced steady results, no major surprises, indicating a stable period."
    text4 = "This stock is going to the moon! 🚀🚀🚀"
    text5 = "Terrible news, the CEO just resigned unexpectedly."

    print(f"'{text1}' -> {analyze_sentiment(text1)}")
    print(f"'{text2}' -> {analyze_sentiment(text2)}")
    print(f"'{text3}' -> {analyze_sentiment(text3)}")
    print(f"'{text4}' -> {analyze_sentiment(text4)}")
    print(f"'{text5}' -> {analyze_sentiment(text5)}")