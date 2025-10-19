import finnhub
import yfinance as yf
from app.utils import config
from datetime import datetime
from pathlib import Path
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
from io import StringIO  # Needed for reading HTML string with pandas

# --- Finnhub Client Initialization ---
try:
    # Ensure config.API_KEY is valid
    if config.API_KEY:
        finnhub_client = finnhub.Client(api_key=config.API_KEY)
        # Optional: Add a test call to verify the key immediately
        # finnhub_client.profile2(symbol='AAPL')
        print("Finnhub client initialized successfully.")
    else:
        finnhub_client = None
        print("Warning: Finnhub API_KEY not found in config. Finnhub features disabled.")
except Exception as e:
    finnhub_client = None
    print(f"Error initializing Finnhub client: {e}")


def get_sp500_data():
    """
    Scrapes StockAnalysis.com for tickers and Wikipedia for founding years.
    Returns: tuple (list_of_tickers, dict_ticker_to_founded_year) or (None, None).
    """
    tickers = []
    ticker_to_founded = {}
    headers = {'User-Agent': 'Mozilla/5.0'}  # Mimic browser

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
        return None, None  # Cannot proceed without tickers

    # --- Step 2: Scrape Founding Years from Wikipedia ---
    print("Fetching founding years from Wikipedia...")
    try:
        url_wiki = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        response_wiki = requests.get(url_wiki, headers=headers, timeout=15)
        response_wiki.raise_for_status()  # Check for HTTP errors (like 403)
        soup = BeautifulSoup(response_wiki.text, 'html.parser')
        table_wiki = soup.find('table', {'id': 'constituents'}) or soup.find('table', {
            'class': 'wikitable sortable'})  # Find table
        if not table_wiki: raise ValueError("Could not find S&P table on Wikipedia")

        wiki_data = {}
        for row in table_wiki.find_all('tr')[1:]:  # Skip header
            columns = row.find_all('td')
            if len(columns) > 7:  # Check expected columns
                try:
                    wiki_ticker = columns[0].text.strip()
                    founded_info = columns[7].text.strip()  # Index 7 = 'Founded' (VERIFY)
                    cleaned_wiki_ticker = str(wiki_ticker).strip().replace('.', '-')
                    match = re.search(r'\b(\d{4})\b', founded_info)  # Extract 4-digit year
                    cleaned_founded = match.group(1) if match else "N/A"
                    wiki_data[cleaned_wiki_ticker] = cleaned_founded
                except IndexError:
                    continue  # Skip row if not enough columns

        # --- Step 3: Merge ---
        for ticker in tickers:
            ticker_to_founded[ticker] = wiki_data.get(ticker, "N/A")  # Default to N/A
        print(f"Matched founding years for {len([y for y in ticker_to_founded.values() if y != 'N/A'])} tickers.")

    except requests.exceptions.RequestException as req_err:
        print(f"HTTP Error fetching Wikipedia: {req_err}. Founding years unavailable.")
        ticker_to_founded = {ticker: "N/A" for ticker in tickers}  # Set all to N/A
    except Exception as e:
        print(f"Error scraping Wikipedia: {e}. Founding years unavailable.")
        ticker_to_founded = {ticker: "N/A" for ticker in tickers}  # Set all to N/A

    return tickers, ticker_to_founded  # Return list and dict


def get_sp500_prices(start="2015-01-01", tickers=None, auto_adjust=True):
    """ Downloads historical OHLCV using yfinance """
    try:
        if tickers is None:
            print("Tickers not provided to get_sp500_prices, fetching using get_sp500_data...")
            # Call the NEW scraping function if tickers aren't provided
            tickers_list, _ = get_sp500_data()  # We only need the list of tickers here
            if not tickers_list:
                print("Failed to fetch tickers list in get_sp500_prices fallback.")
                return None
            tickers = tickers_list  # Use the fetched list

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
    tickers_list, founded_dict = get_sp500_data()

    if tickers_list and founded_dict is not None:  # Check founded_dict existence
        print(f"\nFetched {len(tickers_list)} tickers.")
        print(f"Sample Tickers: {tickers_list[:5]}")
        print("Sample Founding Years (from Wikipedia scrape):")
        for ticker in tickers_list[:5]:
            # Use .get() for safe dictionary access
            print(f" - {ticker}: {founded_dict.get(ticker, 'N/A')}")

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