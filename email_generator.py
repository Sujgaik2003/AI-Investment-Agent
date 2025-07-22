from transformers import pipeline, set_seed
import warnings

# Suppress specific warnings from transformers/pytorch
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint were not used")
warnings.filterwarnings("ignore", message="Some weights of the model did not receive a proper `dot_product`")
warnings.filterwarnings("ignore", message="The model 'GPT2LMHeadModel' is not supported for text-generation. Please use 'CausalLMForSequenceClassification' instead.")


# Global variable to store the text generation pipeline for caching
_text_generation_pipeline = None

def get_text_generation_pipeline():
    """
    Initializes and returns the Hugging Face text generation pipeline (GPT-2).
    """
    global _text_generation_pipeline
    if _text_generation_pipeline is None:
        print("Initializing text generation pipeline (GPT-2)... This might take a moment to download the model.")
        try:
            # Using 'gpt2' model for general text generation.
            # 'distilgpt2' is a smaller, faster alternative if needed.
            _text_generation_pipeline = pipeline(
                "text-generation",
                model="gpt2" # Using gpt2 for now, or 'distilgpt2' for faster loading
            )
            set_seed(42) # For reproducible results
            print("Text generation pipeline initialized.")
        except Exception as e:
            print(f"Error initializing text generation pipeline: {e}")
            print("Please ensure you have an active internet connection to download the model.")
            _text_generation_pipeline = None
    return _text_generation_pipeline

def generate_investment_email(
    ticker: str,
    overall_sentiment_info: dict,
    predictions_summary: str,
    portfolio_advice_summary: str
) -> str:
    """
    Generates a draft investment summary email using an LLM.

    Args:
        ticker (str): The stock ticker symbol.
        overall_sentiment_info (dict): Dictionary with overall sentiment info (label, color, avg_score).
        predictions_summary (str): A summary of the stock price predictions.
        portfolio_advice_summary (str): A summary of the portfolio allocation advice.

    Returns:
        str: The generated email draft.
    """
    pipeline = get_text_generation_pipeline()
    if pipeline is None:
        return "Error: Text generation pipeline not available. Cannot generate email."

    # Construct the prompt for the LLM
    prompt = f"""
    Write a concise investment summary email for a client about {ticker}.
    
    Current stock details:
    - Stock: {ticker}
    - Market Sentiment: {overall_sentiment_info['label']} (Avg Score: {overall_sentiment_info['average_score']:.2f})
    - Price Prediction: {predictions_summary}
    - Portfolio Advice: {portfolio_advice_summary}

    Conclude with a standard disclaimer that this is not financial advice.
    
    Subject: Investment Summary for {ticker}

    Dear Investor,

    Here is a summary of the latest analysis for {ticker}:
    """

    try:
        # Generate text. Adjust max_new_tokens for length control.
        # num_return_sequences=1 ensures single output.
        # pad_token_id and eos_token_id are important for generation quality with GPT-like models.
        # no_repeat_ngram_size prevents repetitive phrases.
        generated_text = pipeline(
            prompt,
            max_new_tokens=300, # Controls the length of the generated text
            num_return_sequences=1,
            do_sample=True,      # Enables sampling for more creative output
            top_k=50,            # Considers top 50 words for sampling
            top_p=0.95,          # Nucleus sampling
            temperature=0.7,     # Controls randomness
            pad_token_id=pipeline.tokenizer.eos_token_id, # Prevents generation from stopping prematurely
            eos_token_id=pipeline.tokenizer.eos_token_id,  # End of sentence token
            no_repeat_ngram_size=3 # Avoids repeating 3-word sequences
        )[0]['generated_text']

        # Extract only the generated part after the prompt
        email_content = generated_text[len(prompt):].strip()

        # Add a more explicit disclaimer at the end just in case LLM misses it
        disclaimer = """
        Please remember that this analysis is for informational purposes only and does not constitute financial advice. Investing involves risks, and past performance is not indicative of future results. Always consult with a qualified financial professional before making any investment decisions.
        """
        
        # Combine subject, greeting from prompt, generated content, and disclaimer
        final_email = f"Subject: Investment Summary for {ticker}\n\nDear Investor,\n\n{email_content}\n\n{disclaimer}\n\nSincerely,\nYour AI Investment Assistant"

        return final_email
    except Exception as e:
        print(f"Error generating email: {e}")
        return "Could not generate email summary. Please check your internet connection and ensure the model is downloaded."

# Example usage (for testing within email_generator.py)
if __name__ == "__main__":
    print("Testing email generator...")
    
    mock_sentiment = {'label': 'Positive', 'color': 'green', 'average_score': 0.85}
    mock_prediction = "The AI model predicts a moderate upward trend for AAPL stock over the next 30 days."
    mock_portfolio_advice = "The optimized portfolio suggests a 45% allocation to SPY, 30% to QQQ, and 25% to BND for a balanced growth strategy."
    
    email_draft = generate_investment_email("AAPL", mock_sentiment, mock_prediction, mock_portfolio_advice)
    print("\n--- GENERATED EMAIL DRAFT ---")
    print(email_draft)
    print("--- END DRAFT ---")