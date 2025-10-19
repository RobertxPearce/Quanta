# scripts/get_data.py

import sys
from pathlib import Path
from app.utils.config import Config
import finnhub
import yfinance as yf
from datetime import datetime
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
from io import StringIO  # Needed for reading HTML string with pandas

# Add project root to path (This helps standalone scripts find 'app')
sys.path.append(str(Path(__file__).resolve().parent.parent))

# --- Finnhub Client Initialization ---
try:
    # Use the API_KEY attribute from the imported Config object
    if Config.API_KEY and Config.API_KEY != 'YOUR_FINNHUB_API_KEY':
        finnhub_client = finnhub.Client(api_key=Config.API_KEY)
        print("Finnhub client initialized successfully.")
    else:
        # Keep client as None if key is missing/placeholder
        finnhub_client = None
        print("Warning: Finnhub API_KEY is missing or using placeholder. Finnhub features disabled.")

except Exception as e:
    # Handle the case where the Config import itself fails
    finnhub_client = None
    if "config" in str(e):
        print("Error initializing Finnhub client: Configuration import failed. Check app/utils/config.py.")
    else:
        print(f"Error initializing Finnhub client: {e}")


def get_sp500_data():
    """
    Scrapes StockAnalysis.com for tickers and Wikipedia for founding years and sectors.
    Returns: tuple (list_of_tickers, dict_ticker_to_founded_year, dict_ticker_to_sector).
    """
    tickers = []
    ticker_to_founded = {}
    ticker_to_sector = {}
    headers = {'User-Agent': 'Mozilla/5.0'}

    # --- Step 1: Scrape Tickers from StockAnalysis.com ---
    print("Fetching S&P 500 tickers from StockAnalysis.com...")
    try:
        url_sa = 'https://stockanalysis.com/list/sp-500-stocks/'
        response_sa = requests.get(url_sa, headers=headers, timeout=15)
        response_sa.raise_for_status()
        tables_sa = pd.read_html(StringIO(response_sa.text))
        if not tables_sa: raise ValueError("No tables found on StockAnalysis page")
        sp500_table_sa = tables_sa[0]
        tickers = sp500_table_sa['Symbol'].tolist() if 'Symbol' in sp500_table_sa.columns else sp500_table_sa.iloc[
            :, 0].tolist()
        tickers = [str(t).strip().replace('.', '-') for t in tickers]
        print(f"Fetched {len(tickers)} tickers from StockAnalysis.com.")
    except Exception as e:
        print(f"Error scraping StockAnalysis.com: {e}")
        # Return all Nones/Empties immediately on critical failure
        return None, None, None  # <-- RETURN 3 ITEMS

    # --- Step 2: Scrape Founding Years and SECTORS from Wikipedia ---
    print("Fetching founding years and sectors from Wikipedia...")
    wiki_data = {}  # Initialize outside the try block

    try:
        url_wiki = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        response_wiki = requests.get(url_wiki, headers=headers, timeout=15)
        response_wiki.raise_for_status()
        soup = BeautifulSoup(response_wiki.text, 'html.parser')
        table_wiki = soup.find('table', {'id': 'constituents'}) or soup.find('table', {'class': 'wikitable sortable'})
        if not table_wiki: raise ValueError("Could not find S&P table on Wikipedia")

        for row in table_wiki.find_all('tr')[1:]:
            columns = row.find_all('td')
            # Check for minimum required columns
            if len(columns) > 7:
                try:
                    wiki_ticker = columns[0].text.strip()
                    cleaned_wiki_ticker = str(wiki_ticker).strip().replace('.', '-')

                    # EXTRACT SECTOR (Index 3 is typically the GICS Sector column)
                    sector_info = columns[3].text.strip()

                    # EXTRACT FOUNDED YEAR (Index 7)
                    founded_info = columns[7].text.strip()
                    match = re.search(r'\b(\d{4})\b', founded_info)
                    cleaned_founded = match.group(1) if match else "N/A"

                    wiki_data[cleaned_wiki_ticker] = {
                        'founded': cleaned_founded,
                        'sector': sector_info.lower().split()[0].replace(',', '').strip()
                    }
                except IndexError:
                    # Skip row if the expected column index is missing (e.g., Index 3 or 7)
                    continue

    except Exception as e:
        # Catch any failure during Step 2 (e.g., HTTP error on Wikipedia)
        print(f"Error during Wikipedia scrape (Step 2): {e}. Proceeding with only data from Step 1.")
        # wiki_data remains empty {} and is handled gracefully in Step 3.

    # --- Step 3: Merge and Consolidate Data (Executed regardless of Step 2's success) ---
    for ticker in tickers:
        # Use wiki_data.get() to safely pull data, defaulting if it wasn't found in Step 2
        data = wiki_data.get(ticker, {'founded': "N/A", 'sector': "unknown"})

        # Populate the final dictionaries
        ticker_to_founded[ticker] = data['founded']
        ticker_to_sector[ticker] = data['sector']

    print(f"Matched founding years for {len([y for y in ticker_to_founded.values() if y != 'N/A'])} tickers.")
    print(f"Fetched sectors for {len([s for s in ticker_to_sector.values() if s != 'unknown'])} tickers.")

    # --- FINAL RETURN ---
    return tickers, ticker_to_founded, ticker_to_sector


def get_sp500_prices(start="2015-01-01", tickers=None, auto_adjust=True):
    """ Downloads historical OHLCV using yfinance """
    try:
        if tickers is None:
            print("Tickers not provided to get_sp500_prices, fetching using get_sp500_data...")
            # Note the function now returns 3 items, so we need to unpack them correctly
            tickers_list, _, _ = get_sp500_data()
            if not tickers_list:
                print("Failed to fetch tickers list in get_sp500_prices fallback.")
                return None
            tickers = tickers_list

        # Ensure tickers list contains only strings
        yahoo_tickers = [str(t) for t in tickers]

        print(f"Downloading price data for {len(yahoo_tickers)} tickers since {start} via yfinance...")
        data = yf.download(yahoo_tickers, start=start, auto_adjust=auto_adjust, group_by="ticker", threads=True)

        if data.empty:
            print("Warning: yfinance returned an empty DataFrame.")
            return None
        # Drop columns where ALL values are NaN (often happens for tickers with no data in period)
        data = data.dropna(axis=1, how='all')
        if data.empty:
            print("Warning: DataFrame became empty after dropping all-NaN columns.")
            return None
        return data
    except Exception as e:
        print(f"Error fetching yfinance price data: {e}")
        return None


# --- get_market_news (Added Fallback) ---
def get_market_news(category='general'):
    """ Fetches general market news from Finnhub. """
    if not finnhub_client:
        print("Error: Finnhub client is not available for market news.")
        return None
    print(f"Fetching news for category: {category}...")
    try:
        # Prefer market_news as it's more standard now
        news_articles = finnhub_client.market_news(category, min_id=0)
        return news_articles
    except AttributeError:
        try:
            # Fallback if general_news exists (older library versions)
            print("Trying finnhub_client.general_news as fallback...")
            news_articles = finnhub_client.general_news(category, min_id=0)
            return news_articles
        except Exception as e_fallback:
            print(f"An error occurred while fetching news (tried market_news & general_news): {e_fallback}")
            return None
    except Exception as e:
        print(f"An error occurred while fetching news with market_news: {e}")
        return None


# --- save_data_to_csv (No Changes Needed) ---
def save_data_to_csv(data: pd.DataFrame, filename: str, directory="data"):
    """ Saves a pandas DataFrame to a CSV file. """
    if data is None or data.empty:
        print("Cannot save empty data.")
        return
    try:
        # Ensure directory exists
        save_path = Path(directory)
        save_path.mkdir(parents=True, exist_ok=True)
        full_path = save_path / filename

        print(f"Saving data to {full_path}...")
        # Use index=True to save the date/timestamp index
        data.to_csv(full_path, index=True)
        print(f"Data successfully saved to {full_path}.")
    except Exception as e:
        print(f"Error saving data to CSV: {e}")


# --- extract_founding_year (Kept for potential other uses, but not primary) ---
def extract_founding_year(summary):
    """ Tries to find 'founded in YYYY' or 'established in YYYY' """
    if not summary or not isinstance(summary, str): return "N/A"
    match = re.search(r"founded in (\d{4})", summary, re.IGNORECASE)
    if match: return match.group(1)
    match = re.search(r"established in (\d{4})", summary, re.IGNORECASE)
    if match: return match.group(1)
    # Add simple check for just a year in context like "since 1980"
    match = re.search(r"(?:since|established|founded)\s+(\d{4})", summary, re.IGNORECASE)
    if match: return match.group(1)
    return "N/A"


# --- _to_yahoo_symbols (Kept just in case) ---
def _to_yahoo_symbols(tickers):
    """ Convert symbols with dots to Yahoo's dash format """
    return [str(t).replace(".", "-") for t in tickers]


def get_latest_csv_path(directory: Path, pattern: str = "*.csv"):
    """ Finds the most recently created CSV file matching a pattern in a directory. """
    try:
        # Check if the directory exists, return None if not
        if not directory.is_dir():
            print(f"Warning: Directory not found at {directory}")
            return None

        # Sort files by last modification time (mtime), descending
        list_of_files = sorted(directory.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)

        if list_of_files:
            return list_of_files[0]
        else:
            print(f"Warning: No files matching '{pattern}' found in {directory}")
            return None
    except Exception as e:
        print(f"Error finding latest CSV: {e}")
        return None


def get_ticker_sectors(tickers: list) -> dict:
    """
    Retrieves sector data from the recently scraped Wikipedia list
    for the provided tickers. (Replaces the Finnhub API call).
    """
    print("Retrieving S&P 500 sector data from scraped cache (API-Free)...")

    # We call the main scraping function to ensure we get the latest sector data.
    # We ignore the founded dict return (_) since we only need the sectors
    tickers_list, _, ticker_to_sector_all = get_sp500_data()

    if not tickers_list:
        return {}  # Scrape failed

    # Return a filtered dictionary containing only the requested tickers
    return {ticker: ticker_to_sector_all.get(ticker, 'unknown') for ticker in tickers}


# --- Main execution block (Updated Test) ---
if __name__ == "__main__":
    # Define parameters
    START_DATE = "2023-01-01"  # Use a more recent start date for faster testing
    DATA_DIR = Path(
        __file__).resolve().parent.parent / "data" / "raw" / "sp500"  # More robust path relative to this script
    BASE_FILENAME = "sp500_prices"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    PRICE_FILENAME = f"{BASE_FILENAME}_{timestamp}.csv"

    print("--- Testing get_sp500_data() ---")
    # Call the combined function
    # Note: Unpack three return values here
    tickers_list, founded_dict, sector_dict = get_sp500_data()

    if tickers_list and founded_dict is not None:  # Check founded_dict existence
        print(f"\nFetched {len(tickers_list)} tickers.")
        print(f"Sample Tickers: {tickers_list[:5]}")
        print("Sample Founding Years (from Wikipedia scrape):")
        for ticker in tickers_list[:5]:
            # Use .get() for safe dictionary access
            print(f" - {ticker}: {founded_dict.get(ticker, 'N/A')}")

        # Print sector data for verification
        print("Sample Sectors (from Wikipedia scrape):")
        for ticker in tickers_list[:5]:
            print(f" - {ticker}: {sector_dict.get(ticker, 'unknown')}")

        print("\n--- Testing get_sp500_prices() ---")
        # Use a smaller sample for the main block test to be faster
        sample_tickers_main = tickers_list[:10]
        prices = get_sp500_prices(tickers=sample_tickers_main, start=START_DATE)

        if prices is not None:
            print("\n--- Price Data Head (Sample) ---")
            # Display head for a couple of tickers only for brevity
            if len(sample_tickers_main) >= 2:
                cols_to_show = [(sample_tickers_main[0], 'Close'), (sample_tickers_main[1], 'Close')]
                # Ensure columns exist before trying to display
                existing_cols = [col for col in cols_to_show if col in prices.columns]
                if existing_cols:
                    print(prices[existing_cols].head())
                else:
                    print("Sample ticker columns not found in downloaded data.")
            else:
                print(prices.head())  # Print full head if less than 2 tickers

                # Uncomment the next line to save the data for the sample
                save_data_to_csv(prices, filename=PRICE_FILENAME, directory=DATA_DIR)
        else:
            print("Failed to fetch price data in main block.")

    else:
        print("Failed to fetch S&P 500 data (tickers and/or founded years).")