# app/ml/features.py
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


from scripts.get_data import get_latest_csv_path, save_data_to_csv
from scripts.StockSentimentMapper import StockSentimentMapper  # adjust import if your class lives elsewhere

BASE_DIR      = Path(__file__).resolve().parents[2]  # repo root
PRICE_DATA_DIR= BASE_DIR / "Quanta" / "data" / "raw" / "sp500"
NEWS_DATA_DIR = BASE_DIR / "Quanta" / "data" / "raw" / "news"
ML_DATA_DIR   = BASE_DIR / "Quanta" / "data" / "processed"

def _safe_pct_change(s: pd.Series, periods: int = 1) -> pd.Series:
    s = s.astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = s.pct_change(periods=periods)
    return out.replace([np.inf, -np.inf], np.nan)

def generate_ml_training_dataset(ticker: str, lookback_days: int = 365) -> Optional[pd.DataFrame]:
    """
    Create a supervised ML dataset by aligning (1) close prices and (2) daily news sentiment.
    Target is next-day percent change. Includes simple engineered features.

    Features:
      - daily_sentiment (mean of mapped headline sentiment per day)
      - sentiment_3d / sentiment_7d (rolling means)
      - ret_1d / ret_5d / ret_20d (historic pct changes)
      - vol_20d (20-day rolling volatility of returns)
      - Close_Price_T (current close to provide scale)
      - Sector_Feature (categorical code), Sector_Name

    Returns:
      DataFrame indexed by date with all features + Target_Pct_Change.
    """
    print(f"\n--- Generating ML dataset for {ticker} ---")

    # 1) Load most-recent prices (expects “sp500_prices_web_pull_*.csv” saved explicitly via ?save=1)
    price_path = get_latest_csv_path(PRICE_DATA_DIR, "sp500_prices_web_pull_*.csv")
    if not price_path:
        print("❌ No raw price CSV found in data/raw/sp500/. Opt-in save from /predict with ?save=1 first.")
        return None

    price_df = pd.read_csv(price_path, index_col=0, header=[0, 1], parse_dates=True)
    price_df.index.name = "Date"

    price_key = (ticker, "Close")
    if price_key not in price_df.columns:
        print(f"❌ Ticker '{ticker}' not present in latest price CSV.")
        return None

    close = price_df[price_key].dropna().rename("Close")
    # restrict to lookback window
    start = datetime.now() - timedelta(days=lookback_days)
    close = close.loc[close.index >= start]

    if close.empty:
        print(f"⚠️ No close prices for {ticker} in the last {lookback_days} days.")
        return None

    # Targets and basic price-derived features
    ret_1d  = _safe_pct_change(close, periods=1).rename("ret_1d")
    ret_5d  = _safe_pct_change(close, periods=5).rename("ret_5d")
    ret_20d = _safe_pct_change(close, periods=20).rename("ret_20d")
    vol_20d = ret_1d.rolling(20, min_periods=10).std().rename("vol_20d")
    target  = ret_1d.shift(-1).rename("Target_Pct_Change")  # predict next-day move

    # 2) Load most-recent news (expects “market_news_sentiment_*.csv” saved explicitly via ?save=1)
    news_path = get_latest_csv_path(NEWS_DATA_DIR, "market_news_sentiment_*.csv")
    if not news_path:
        print("⚠️ No news CSV found in data/raw/news/. Opt-in save from / with ?save=1 to create one.")
        return None

    news_df = pd.read_csv(news_path, index_col=0, parse_dates=True)
    # Normalize dates to daily bins
    news_df.index = pd.to_datetime(news_df.index, errors='coerce').normalize()
    news_df = news_df[news_df.index.notna()]

    # 3) Map sentiment with your custom mapper
    mapper = StockSentimentMapper([ticker])
    sector_name = mapper.stock_sector_mapping.get(ticker, "unknown")

    # raw sentiment per headline (your mapper’s internal method)
    news_df["raw_sentiment_score"] = news_df["headline"].astype(str).apply(mapper._calculate_raw_sentiment)

    # Aggregate to daily
    daily_sent = (
        news_df.groupby(news_df.index)["raw_sentiment_score"]
        .mean()
        .to_frame("daily_sentiment")
        .sort_index()
    )

    # rolling sentiment features
    daily_sent["sentiment_3d"] = daily_sent["daily_sentiment"].rolling(3, min_periods=1).mean()
    daily_sent["sentiment_7d"] = daily_sent["daily_sentiment"].rolling(7, min_periods=3).mean()

    # 4) Merge features
    feat = pd.concat(
        [daily_sent, close.rename("Close_Price_T"), ret_1d, ret_5d, ret_20d, vol_20d, target],
        axis=1
    ).dropna(how="all")

    # keep rows where target is defined
    feat = feat.loc[feat["Target_Pct_Change"].notna()]

    # sector categorical
    feat["Sector_Name"] = sector_name
    feat["Sector_Feature"] = pd.Categorical(feat["Sector_Name"]).codes

    # 5) Save processed dataset (explicit job, not page load)
    ML_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_name = f"ml_data_{ticker}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    save_data_to_csv(feat, out_name, ML_DATA_DIR)
    print(f"✅ Saved ML dataset → {ML_DATA_DIR / out_name} ({len(feat)} rows)")

    return feat


if __name__ == "__main__":

    print("\n--- Starting Batch ML Dataset Generation ---")

    # 1. Find the latest raw price file
    price_path = get_latest_csv_path(PRICE_DATA_DIR, "sp500_prices_web_pull_*.csv")

    if not price_path:
        print("❌ CRITICAL: No raw price CSV found. Please run the /predict route with ?save=1 first.")
        # Attempt to run generate_ml_training_dataset with a sample ticker just to show configuration error if any
        generate_ml_training_dataset("MSFT")
        sys.exit(1)

    print(f"✅ Found raw price data: {price_path.name}")

    try:
        # 2. Load the file just to get the list of tickers
        # Note: We expect header=[0, 1] for the MultiIndex
        price_df_header = pd.read_csv(price_path, index_col=0, header=[0, 1], nrows=1)

        # Extract the unique tickers from the first level of the MultiIndex columns
        all_tickers = sorted(list(set(price_df_header.columns.get_level_values(0).tolist())))

        # Filter out multi-column tickers if necessary (like 'MSFT')
        # We only want the top-level ticker symbol.
        unique_tickers = [t for t in all_tickers if t != '']

    except Exception as e:
        print(f"❌ FAILED to load/parse ticker list from CSV: {e}")
        sys.exit(1)

    print(f"\nFound {len(unique_tickers)} tickers to process.")
    print("--------------------------------------------------")

    success_count = 0
    failure_list = []

    # 3. Loop through all found tickers and generate the dataset
    for i, ticker in enumerate(unique_tickers):
        print(f"\n[{i + 1}/{len(unique_tickers)}] Processing {ticker}...")

        # The generate_ml_training_dataset function handles all loading,
        # filtering, feature engineering, and saving for this single ticker.
        try:
            result_df = generate_ml_training_dataset(ticker)
            if result_df is not None and not result_df.empty:
                success_count += 1
        except Exception as e:
            failure_list.append(ticker)
            print(f"❌ Failed to process {ticker} due to error: {e}")

    print("\n\n--- Batch Summary ---")
    print(f"Total Tickers Processed: {len(unique_tickers)}")
    print(f"Successful Saves: {success_count} ✅")
    if failure_list:
        print(f"Failures: {len(failure_list)} ❌ (See logs for details)")