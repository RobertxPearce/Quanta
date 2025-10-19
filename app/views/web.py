"""
Handles all web page routes for Quanta
"""
from flask import Blueprint, render_template, jsonify, request
from app.ml.infer import run_inference

from scripts.polarity_analysis import get_sentiment_score
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

# Import Path for robust path handling
from pathlib import Path


web = Blueprint('web', __name__)

@web.route('/')
def index():
    """
    Render the homepage (index.html), fetch market news, calculate sentiment,
    and save the results to a CSV file for training.
    """

    from scripts.get_data import get_market_news, save_data_to_csv

    # --- 1. Define paths for saving news data ---
    NEWS_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "news"
    NEWS_BASE_FILENAME = "market_news_sentiment"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    NEWS_FILENAME = f"{NEWS_BASE_FILENAME}_{timestamp}.csv"

    news_list = get_market_news('general')

    if news_list:
        # --- 2. Calculate Sentiment and Store ---
        for article in news_list:
            # article is a dict containing 'headline', 'summary', 'url', 'datetime', etc.
            text_to_analyze = article['headline'] + " " + article['summary']
            # Add the sentiment score directly to the article dictionary
            article['sentiment'] = get_sentiment_score(text_to_analyze)

        # --- 3. Convert List of Dictionaries to DataFrame ---
        # The list contains all the necessary fields, including the new 'sentiment'
        news_df = pd.DataFrame(news_list)

        # --- 4. Select/Rename relevant columns for ML and Save ---
        # Keep relevant columns like 'headline', 'summary', 'datetime', and 'sentiment'
        # Convert Finnhub timestamp (seconds) to datetime
        if 'datetime' in news_df.columns:
            news_df['datetime'] = pd.to_datetime(news_df['datetime'], unit='s')
            news_df.set_index('datetime', inplace=True)  # Use datetime as the index

        # Explicitly select columns you want to save
        columns_to_save = ['headline', 'summary', 'source', 'sentiment']
        news_df_to_save = news_df[[col for col in columns_to_save if col in news_df.columns]]

        # Call the saving function
        save_data_to_csv(news_df_to_save, filename=NEWS_FILENAME, directory=NEWS_DATA_DIR)

    return render_template("index.html", title='Home', articles=news_list)

@web.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Fetches S&P 500 tickers and founding years, calculates growth for a sample,
    and displays the top 5 growth stocks.
    """

    from scripts.get_data import get_sp500_data, get_sp500_prices, save_data_to_csv

    tickers_list = []  # Full list from scrape
    founded_dict = {}  # Dict {ticker: year} from scrape
    price_table_html = None  # Optional: HTML table of recent prices
    sample_tickers_used = []  # Subset used for price fetching/growth calc
    top_5_growth = []  # Result: [('TICKER', growth_pct, current_price), ...]

    growth_data_all = {}  # {ticker: growth_pct}
    current_prices_all = {}  # {ticker: price}

    # --- NEW: Define paths and filename for saving data ---
    # Define a base directory for raw data relative to the project root (assuming 'web.py' is in 'app')
    # Adjust this path as necessary for your project structure
    # This path setup assumes 'web.py' is in a structure like: project_root/app/web.py
    # and you want to save to: project_root/data/raw/sp500/
    # If the app structure is simple (e.g., all files in one folder), you can simplify `DATA_DIR`
    DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "sp500"
    BASE_FILENAME = "sp500_prices_web_pull"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    PRICE_FILENAME = f"{BASE_FILENAME}_{timestamp}.csv"

    print("Fetching S&P data (Tickers from SA, Founded from Wiki)...")
    tickers_list, founded_dict, sector_dict = get_sp500_data()

    if not tickers_list:
        print("Error: Failed to fetch S&P ticker list. Cannot proceed.")
        return render_template('predict.html',
                               # ... (omitted for brevity - error return) ...
                               title='S&P 500 Data Error',
                               tickers=[],
                               founding_years={},
                               top_growth_stocks=[],
                               price_table="<p>Error fetching S&P 500 constituents.</p>")

    founded_dict = founded_dict or {}
    sample_tickers_used = tickers_list  # Use all tickers

    one_year_ago_plus_buffer = (datetime.now() - timedelta(days=370)).strftime('%Y-%m-%d')
    print(f"Fetching price data for {len(sample_tickers_used)} tickers since {one_year_ago_plus_buffer}...")
    prices_df = get_sp500_prices(tickers=sample_tickers_used, start=one_year_ago_plus_buffer)

    if prices_df is not None and not prices_df.empty:

        # --- NEW INTEGRATION: Save the DataFrame to CSV before processing ---
        # The save_data_to_csv function is designed to handle this robustly.
        save_data_to_csv(prices_df, filename=PRICE_FILENAME, directory=DATA_DIR)

        if isinstance(prices_df.columns, pd.MultiIndex):
            one_year_ago_dt = datetime.now() - timedelta(days=365)
            # ... (omitted for brevity - growth calculation logic) ...

            for ticker in sample_tickers_used:
                if (ticker, 'Close') in prices_df.columns:
                    ticker_close_adjusted = prices_df[(ticker, 'Close')].dropna()
                    valid_prices = ticker_close_adjusted[ticker_close_adjusted.index >= pd.to_datetime(one_year_ago_dt)]

                    if len(valid_prices) > 1:
                        start_price = valid_prices.iloc[0]
                        end_price = valid_prices.iloc[-1]  # This is the current price

                        current_prices_all[ticker] = round(end_price, 2)

                        if start_price and pd.notna(start_price) and start_price != 0:
                            growth_pct = ((end_price - start_price) / start_price) * 100
                            growth_data_all[ticker] = round(growth_pct, 2)
                        else:
                            growth_data_all[ticker] = 0.0
                    else:
                        growth_data_all[ticker] = 0.0
                        current_prices_all[ticker] = None
                else:
                    print(f"Warning: Column ('{ticker}', 'Close') not found for growth calc.")
                    growth_data_all[ticker] = 0.0
                    current_prices_all[ticker] = None

            # ... (omitted for brevity - top 5 and HTML table generation) ...
            sorted_growth_list = sorted(growth_data_all.items(), key=lambda item: item[1], reverse=True)

            top_5_growth = []  # Reset to ensure it's the new format
            for ticker, growth in sorted_growth_list[:5]:
                current_price = current_prices_all.get(ticker)
                top_5_growth.append((ticker, growth, current_price))

            # ... (HTML Table Generation - unchanged) ...
            try:
                columns_to_show = [(t, col) for t in sample_tickers_used[:20] for col in ['Open', 'Close', 'Volume'] if
                                   (t, col) in prices_df.columns]
                if columns_to_show:
                    price_table_html = prices_df[columns_to_show].tail().to_html(
                        classes='table table-striped table-sm table-hover', border=0)
                else:
                    price_table_html = "<p>No valid price columns found for sample.</p>"
            except Exception as table_error:
                print(f"Error converting prices_df to HTML: {table_error}")
                price_table_html = "<p>Error displaying price table.</p>"


        else:
            print("Warning: prices_df columns not MultiIndex. Cannot calculate growth.")

    else:
        price_table_html = "<p>Could not fetch price data for sample tickers.</p>"
        top_5_growth = []

    return render_template('predict.html',
                           title='S&P 500 Analysis',
                           tickers=tickers_list,
                           founding_years=founded_dict,
                           sample_tickers=sample_tickers_used,
                           top_growth_stocks=top_5_growth,
                           price_table=price_table_html,
                           growth_data=growth_data_all,
                           current_prices=current_prices_all)