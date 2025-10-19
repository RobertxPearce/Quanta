import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import yfinance as yf
import re
import requests
from bs4 import BeautifulSoup
from io import StringIO
import finnhub
from typing import Union, Optional

from app.utils.config import Config

sys.path.append(str(Path(__file__).resolve().parent.parent))

# --- Initialize Finnhub Client ---
try:
    finnhub_client = (
        finnhub.Client(api_key=Config.API_KEY)
        if Config.API_KEY not in {None, "YOUR_FINNHUB_API_KEY"}
        else None
    )
    print("✅ Finnhub client initialized." if finnhub_client else "⚠️ Finnhub API_KEY missing or placeholder.")
except Exception as e:
    finnhub_client = None
    print(f"⚠️ Finnhub init failed: {e}")


# --- Core Data Functions ---
def get_sp500_data():
    """Scrapes S&P 500 tickers, sectors, and founding years."""
    headers = {"User-Agent": "Mozilla/5.0"}
    tickers, ticker_to_founded, ticker_to_sector = [], {}, {}

    # StockAnalysis.com Scrape
    try:
        print("Fetching S&P 500 tickers...")
        sa_resp = requests.get("https://stockanalysis.com/list/sp-500-stocks/", headers=headers, timeout=15)
        sa_resp.raise_for_status()
        tickers = pd.read_html(StringIO(sa_resp.text))[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
    except Exception as e:
        print(f"❌ Error scraping tickers: {e}")
        return None, None, None

    # Wikipedia Scrape for sectors/founding
    try:
        print("Fetching sectors and founding years...")
        wiki_resp = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=headers,
            timeout=15,
        )
        wiki_resp.raise_for_status()
        soup = BeautifulSoup(wiki_resp.text, "html.parser")
        table = soup.find("table", {"id": "constituents"}) or soup.find("table", {"class": "wikitable sortable"})

        for row in table.find_all("tr")[1:]:
            cols = row.find_all("td")
            if len(cols) > 7:
                ticker = cols[0].text.strip().replace(".", "-")
                sector = cols[3].text.strip().lower().split()[0]
                founded = re.search(r"\b(\d{4})\b", cols[7].text.strip())
                ticker_to_sector[ticker] = sector
                ticker_to_founded[ticker] = founded.group(1) if founded else "N/A"
    except Exception as e:
        print(f"⚠️ Wikipedia scrape failed: {e}")

    for ticker in tickers:
        ticker_to_sector.setdefault(ticker, "unknown")
        ticker_to_founded.setdefault(ticker, "N/A")

    return tickers, ticker_to_founded, ticker_to_sector


def get_sp500_prices(start="2015-01-01", tickers=None, auto_adjust=True):
    """Fetch historical price data using yfinance."""
    try:
        if tickers is None:
            tickers, _, _ = get_sp500_data()
            if not tickers:
                return None

        data = yf.download(tickers, start=start, auto_adjust=auto_adjust, group_by="ticker", threads=True)
        if data.empty:
            print("⚠️ Empty DataFrame from yfinance.")
            return None
        return data.dropna(axis=1, how="all")
    except Exception as e:
        print(f"❌ Error fetching prices: {e}")
        return None


def get_market_news(category="general"):
    """Fetch market news via Finnhub (if API key available)."""
    if not finnhub_client:
        print("⚠️ Finnhub unavailable.")
        return None
    try:
        # Use the correct API method for your finnhub version
        return finnhub_client.general_news(category=category, min_id=0)
    except AttributeError:
        print("⚠️ Falling back: 'general_news' not available in this Finnhub client.")
        return None
    except Exception as e:
        print(f"❌ News fetch failed: {e}")
        return None



def save_data_to_csv(data: pd.DataFrame, filename: str, directory: Union[str, Path]):
    """Save DataFrame to CSV."""
    if data is None or data.empty:
        print("⚠️ Skipping empty dataset save.")
        return
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    data.to_csv(path)
    print(f"💾 Saved: {path}")


def get_latest_csv_path(directory: Path, pattern: str = "*.csv") -> Optional[Path]:
    """Return the latest CSV file matching a pattern."""
    try:
        if not directory.is_dir():
            print(f"⚠️ Directory not found: {directory}")
            return None
        files = sorted(directory.glob(pattern), key=lambda f: f.stat().st_mtime, reverse=True)
        return files[0] if files else None
    except Exception as e:
        print(f"❌ Error finding CSV: {e}")
        return None


def get_ticker_sectors(tickers: list) -> dict:
    """Get sectors for specified tickers via cached scrape."""
    print("Fetching sectors (scrape cache)...")
    tickers_all, _, sectors_all = get_sp500_data()
    if not tickers_all:
        return {}
    return {t: sectors_all.get(t, "unknown") for t in tickers}


if __name__ == "__main__":
    tickers, _, sectors = get_sp500_data()
    if tickers:
        print(f"✅ Retrieved {len(tickers)} tickers. Example: {tickers[:5]}")
        prices = get_sp500_prices(tickers=tickers[:10], start="2023-01-01")
        if prices is not None:
            save_data_to_csv(prices, "sp500_prices_test.csv", Path(__file__).resolve().parent / "data" / "raw")
