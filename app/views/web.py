"""
Handles all web page routes for Quanta
"""
from flask import Blueprint, render_template, request
from app.ml.infer import run_inference
from scripts.polarity_analysis import get_sentiment_score
from datetime import datetime, timedelta
from pathlib import Path
import os
import pandas as pd
import yfinance as yf

web = Blueprint('web', __name__)

def _should_persist() -> bool:
    """
    Returns True only if you explicitly allow writes.
    - via query param:  ?save=1
    - or env var:       QUANTA_ALLOW_WRITE=1
    """
    return (request.args.get("save") == "1") or (os.getenv("QUANTA_ALLOW_WRITE") == "1")

@web.route('/')
def index():
    """
    Render the homepage (index.html), fetch market news, calculate sentiment.
    NOTE: No files are written on page load. To allow saving, call with ?save=1.
    """
    from scripts.get_data import get_market_news, save_data_to_csv  # local import to keep views light

    news_list = get_market_news('general')

    if news_list:
        for article in news_list:
            # article: dict with 'headline', 'summary', 'url', 'datetime', etc.
            text_to_analyze = f"{article.get('headline','')} {article.get('summary','')}"
            article['sentiment'] = get_sentiment_score(text_to_analyze)

        # Optional (explicitly requested) persistence for offline training pipelines
        if _should_persist():
            NEWS_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "news"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"market_news_sentiment_{timestamp}.csv"

            news_df = pd.DataFrame(news_list)
            if 'datetime' in news_df.columns:
                news_df['datetime'] = pd.to_datetime(news_df['datetime'], unit='s', errors='coerce')
                news_df.set_index('datetime', inplace=True)

            cols = [c for c in ['headline','summary','source','sentiment'] if c in news_df.columns]
            save_data_to_csv(news_df[cols] if cols else news_df, filename=filename, directory=NEWS_DATA_DIR)

    return render_template("index.html", title='Home', articles=news_list)

@web.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Fetches S&P 500 tickers and founding years, calculates growth for a sample,
    and displays the top 5 growth stocks.
    NOTE: No files are written on page load. To allow saving, call with ?save=1.
    """
    from scripts.get_data import get_sp500_data, get_sp500_prices, save_data_to_csv

    tickers_list, founded_dict, sector_dict = get_sp500_data()
    if not tickers_list:
        return render_template(
            'predict.html',
            title='S&P 500 Data Error',
            tickers=[],
            founding_years={},
            top_growth_stocks=[],
            price_table="<p>Error fetching S&P 500 constituents.</p>",
            growth_data={},
            current_prices={}
        )

    founded_dict = founded_dict or {}
    sample_tickers_used = tickers_list  # use all

    one_year_ago_plus_buffer = (datetime.now() - timedelta(days=370)).strftime('%Y-%m-%d')
    prices_df = get_sp500_prices(tickers=sample_tickers_used, start=one_year_ago_plus_buffer)

    growth_data_all, current_prices_all = {}, {}
    price_table_html, top_5_growth = None, []

    if prices_df is not None and not prices_df.empty:
        # Only persist if explicitly allowed
        if _should_persist():
            DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "sp500"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sp500_prices_web_pull_{timestamp}.csv"
            save_data_to_csv(prices_df, filename=filename, directory=DATA_DIR)

        if isinstance(prices_df.columns, pd.MultiIndex):
            one_year_ago_dt = datetime.now() - timedelta(days=365)

            for ticker in sample_tickers_used:
                key = (ticker, 'Close')
                if key in prices_df.columns:
                    series = prices_df[key].dropna()
                    valid = series[series.index >= pd.to_datetime(one_year_ago_dt)]
                    if len(valid) > 1:
                        start_price = valid.iloc[0]
                        end_price = valid.iloc[-1]
                        current_prices_all[ticker] = round(float(end_price), 2)
                        growth_data_all[ticker] = round(((end_price - start_price) / start_price) * 100, 2) if start_price else 0.0
                    else:
                        growth_data_all[ticker] = 0.0
                        current_prices_all[ticker] = None
                else:
                    growth_data_all[ticker] = 0.0
                    current_prices_all[ticker] = None

            sorted_growth = sorted(growth_data_all.items(), key=lambda kv: kv[1], reverse=True)
            top_5_growth = [(t, g, current_prices_all.get(t)) for t, g in sorted_growth[:5]]

            try:
                sample_cols = [(t, c) for t in sample_tickers_used[:20] for c in ['Open','Close','Volume'] if (t, c) in prices_df.columns]
                price_table_html = prices_df[sample_cols].tail().to_html(classes='table table-striped table-sm table-hover', border=0) if sample_cols else "<p>No valid price columns found for sample.</p>"
            except Exception:
                price_table_html = "<p>Error displaying price table.</p>"
        else:
            # columns not MultiIndex
            pass
    else:
        price_table_html = "<p>Could not fetch price data for sample tickers.</p>"
        top_5_growth = []

    return render_template(
        'predict.html',
        title='S&P 500 Analysis',
        tickers=tickers_list,
        founding_years=founded_dict,
        sample_tickers=sample_tickers_used,
        top_growth_stocks=top_5_growth,
        price_table=price_table_html,
        growth_data=growth_data_all,
        current_prices=current_prices_all
    )
