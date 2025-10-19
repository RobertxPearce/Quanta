import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

from .get_data import get_latest_csv_path, save_data_to_csv
from .StockSentimentMapper import StockSentimentMapper
from app.utils.config import Config


BASE_DIR = Path(__file__).resolve().parent.parent
PRICE_DATA_DIR = BASE_DIR / "data" / "raw" / "sp500"
NEWS_DATA_DIR = BASE_DIR / "data" / "raw" / "news"
ML_DATA_DIR = BASE_DIR / "data" / "processed"


def generate_ml_training_dataset(ticker: str, price_lookback_days: int = 365) -> pd.DataFrame | None:
    """
    Generate a machine learning dataset aligning stock prices and news sentiment.

    Args:
        ticker (str): Stock ticker symbol.
        price_lookback_days (int): Number of past days to include.

    Returns:
        pd.DataFrame | None: Dataset ready for ML model training.
    """
    print(f"\n--- Generating ML Dataset for {ticker} ---")

    # 1. Load Latest Price Data
    latest_price_path = get_latest_csv_path(PRICE_DATA_DIR, "sp500_prices_web_pull_*.csv")
    if not latest_price_path:
        print("❌ ERROR: Price CSV not found. Run the /predict route first.")
        return None

    price_df = pd.read_csv(latest_price_path, index_col=0, header=[0, 1], parse_dates=True)
    price_df.index.name = "Date"

    if (ticker, "Close") not in price_df.columns:
        print(f"❌ ERROR: Ticker {ticker} missing from price data.")
        return None

    close_prices = price_df[(ticker, "Close")].dropna()
    target_y = close_prices.pct_change().shift(-1).dropna().to_frame("Target_Pct_Change")

    # Filter to lookback period
    start_date = datetime.now() - timedelta(days=price_lookback_days)
    target_y = target_y.loc[target_y.index >= start_date]

    if target_y.empty:
        print(f"⚠️ No price change data for {ticker} in last {price_lookback_days} days.")
        return None

    # 2. Load Latest News Data
    latest_news_path = get_latest_csv_path(NEWS_DATA_DIR, "market_news_sentiment_*.csv")
    if not latest_news_path:
        print("⚠️ WARNING: News CSV not found. Cannot generate sentiment features.")
        return None

    news_df = pd.read_csv(latest_news_path, index_col=0, parse_dates=True)
    news_df.index = news_df.index.normalize()

    # 3. Sentiment Mapping
    mapper = StockSentimentMapper([ticker])
    sector = mapper.stock_sector_mapping.get(ticker, "unknown")
    print(f"ℹ️ {ticker} sector: {sector}")

    news_df["raw_sentiment_score"] = news_df["headline"].apply(mapper._calculate_raw_sentiment)
    daily_sentiment = news_df.groupby(news_df.index)["raw_sentiment_score"].mean().to_frame("daily_sentiment")

    # 4. Merge Sentiment with Price Data
    combined_df = (
        daily_sentiment.merge(target_y, left_index=True, right_index=True, how="inner")
        .merge(close_prices.to_frame("Close_Price_T"), left_index=True, right_index=True)
    )
    combined_df["Sector_Name"] = sector
    combined_df["Sector_Feature"] = pd.Categorical(combined_df["Sector_Name"]).codes

    # 5. Save Final Dataset
    ML_DATA_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"ml_data_{ticker}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    save_data_to_csv(combined_df, filename, ML_DATA_DIR)

    print(f"✅ SUCCESS: {len(combined_df)} records saved → {ML_DATA_DIR / filename}")
    return combined_df


if __name__ == "__main__":
    df = generate_ml_training_dataset("MSFT")
    if df is not None:
        print("\n--- Sample of Final ML Training Dataset ---")
        print(df.head())
