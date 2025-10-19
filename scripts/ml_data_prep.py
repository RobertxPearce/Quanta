# scripts/ml_prep.py

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np


from .get_data import get_latest_csv_path, save_data_to_csv
from .StockSentimentMapper import StockSentimentMapper
from app.utils.config import Config

# Import helper functions from your existing files
from .get_data import get_latest_csv_path, save_data_to_csv
from .StockSentimentMapper import StockSentimentMapper

BASE_DIR = Path(__file__).resolve().parent.parent
# if you are running project root is Quanta/.

PRICE_DATA_DIR = BASE_DIR / "data" / "raw" / "sp500"
NEWS_DATA_DIR = BASE_DIR / "data" / "raw" / "news"
ML_DATA_DIR = BASE_DIR / "data" / "processed"


def generate_ml_training_dataset(ticker: str, price_lookback_days: int = 365) -> pd.DataFrame or None:
    """
    Loads the latest saved price and news data, aligns them, extracts
    sentiment features, and generates the final ML training dataset.

    Args:
        ticker (str): The specific ticker to build the dataset for.
        price_lookback_days (int): Days of price data to consider for target Y.

    Returns:
        pd.DataFrame or None: The combined dataset ready for ML training.
    """

    print(f"\n--- Generating ML Dataset for Ticker: {ticker} ---")

    # 1. Load the Latest Saved Price Data
    latest_price_path = get_latest_csv_path(PRICE_DATA_DIR, pattern="sp500_prices_web_pull_*.csv")
    if not latest_price_path:
        print("ERROR: Could not find latest price CSV. Run the /predict route first.")
        return None

    price_df = pd.read_csv(latest_price_path, index_col=0, header=[0, 1], parse_dates=True)
    price_df.index.name = 'Date'

    # 2. Extract Price Target (Y)
    if (ticker, 'Close') not in price_df.columns:
        print(f"ERROR: Ticker {ticker} not found in price data columns.")
        return None

    close_prices = price_df[(ticker, 'Close')].dropna()

    # Calculate the next day's percentage change (the target Y)
    # News on day T predicts the price change from day T to T+1.
    target_y = close_prices.pct_change().shift(-1).dropna()

    target_y_df = target_y.to_frame('Target_Pct_Change')

    start_date = datetime.now() - timedelta(days=price_lookback_days)
    target_y = target_y[target_y.index >= start_date.strftime('%Y-%m-%d')]

    if target_y.empty:
        print(f"Warning: No valid price change data for {ticker} in the last {price_lookback_days} days.")
        return None

    # 3. Load the Latest Saved News Data (X features)
    latest_news_path = get_latest_csv_path(NEWS_DATA_DIR, pattern="market_news_sentiment_*.csv")
    if not latest_news_path:
        print("WARNING: Could not find latest news CSV. Cannot generate sentiment features.")
        return None

    news_df = pd.read_csv(latest_news_path, index_col=0, parse_dates=True)
    news_df.index = news_df.index.normalize()  # Remove time component for merging
    news_df.index.name = 'Date'

    # 4. Apply StockSentimentMapper and Prepare Features
    # Initialize mapper, triggering the API call for the ticker's sector
    mapper = StockSentimentMapper(tickers_to_analyze=[ticker])

    # Get the sector for the current ticker
    current_ticker_sector = mapper.stock_sector_mapping.get(ticker, 'unknown')
    print(f"INFO: {ticker} sector (from API): {current_ticker_sector}")

    # Apply raw sentiment score using the mapper's internal VADER call
    news_df['raw_sentiment_score'] = news_df['headline'].apply(mapper._calculate_raw_sentiment)

    # Group news data by date and average the sentiment for a daily signal
    daily_sentiment = news_df.groupby(news_df.index)['raw_sentiment_score'].mean().to_frame('daily_sentiment')

    # 5. Combine Data
    # Merge the daily sentiment (X) with the price change target (Y)
    combined_df = daily_sentiment.merge(
        target_y_df,
        left_index=True,
        right_index=True,
        how='inner'
    )

    # Add the current closing price (as a feature X)
    current_close_price = close_prices.loc[combined_df.index].to_frame('Close_Price_T')
    combined_df = combined_df.merge(current_close_price, left_index=True, right_index=True)

    combined_df['Sector_Name'] = current_ticker_sector

    # Add the sector as a feature (converted to a simple integer for basic ML)
    #combined_df['Sector_Feature'] = pd.Categorical(combined_df.index.map(lambda x: current_ticker_sector)).codes
    combined_df['Sector_Feature'] = pd.Categorical(combined_df['Sector_Name']).codes

    # 6. Save the Final Dataset
    ML_DATA_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ml_data_{ticker}_{timestamp}.csv"

    save_data_to_csv(combined_df, filename=filename, directory=ML_DATA_DIR)

    print(f"SUCCESS: Generated ML dataset with {len(combined_df)} records.")
    print(f"Saved to: {ML_DATA_DIR / filename}")
    return combined_df


# --- Example Usage (Run this from your main project folder) ---
if __name__ == '__main__':
    TICKER_TO_ANALYZE = 'MSFT'
    ml_dataset = generate_ml_training_dataset(TICKER_TO_ANALYZE)

    if ml_dataset is not None:
        print("\n--- Sample of Final ML Training Dataset ---")
        print(ml_dataset.head())