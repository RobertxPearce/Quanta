import finnhub
from app.utils import config

try:
    finnhub_client = finnhub.Client(api_key=config.API_KEY)
    print("Finnhub client initialized successfully.")
except Exception as e:
    finnhub_client = None


def get_market_news(category='general'):
    """
    Fetches general market news from Finnhub.
    Args: category (str): The category to fetch (e.g., 'general', 'forex', 'crypto').
    Returns: list: A list of news articles, or None if an error occurs.
    """

    # Don't proceed if the client failed to initialize
    if not finnhub_client:
        print("Error: Finnhub client is not available.")
        return None

    print(f"Fetching news for category: {category}...")
    try:
        news_articles = finnhub_client.general_news('general', min_id=0)
        return news_articles

    except Exception as e:
        print(f"An error occurred while fetching news: {e}")
        return None

if __name__ == "__main__":
    # One-off testing or CLI here
    # print("Finnhub client initialized successfully.")
    # init_client(); print(test_fetch())
    pass