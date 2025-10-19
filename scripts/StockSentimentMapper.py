# scripts/StockSentimentMapper.py

# Import the specific numerical function from your file
from .polarity_analysis import get_compound_score
from .get_data import get_ticker_sectors


class StockSentimentMapper:
    """
    A utility class to extract nuanced sentiment features (X variables)
    from news headlines, considering sector and keyword impact, for
    Machine Learning training against stock price changes (Y variable).
    """

    def __init__(self, tickers_to_analyze: list = None):  # <-- ADDED ARGUMENT
        # Predefined sentiment and sector impact mappings
        self.sector_sentiment_map = {
            'defense': {
                'war': 'positive',
                'conflict': 'positive',
                'military_spending': 'positive'
            },
            'tech': {
                'war': 'negative',
                'cybersecurity_threat': 'negative',
                'chip_shortage': 'negative'
            },
            'healthcare': {
                'pandemic': 'complex',
                'medical_breakthrough': 'positive'
            }
        }

        # --- API FETCH INTEGRATION ---
        if tickers_to_analyze:
            # Fetch sectors for the list of tickers passed in via the API call
            self.stock_sector_mapping = get_ticker_sectors(tickers_to_analyze)
        else:
            # Keep a small, hardcoded map as a fallback
            self.stock_sector_mapping = {
                'LMT': 'defense',
                'MSFT': 'tech',
                'GOOGL': 'tech',
                'BA': 'defense',
                'NVDA': 'tech',
                'AAPL': 'tech',
                'JNJ': 'healthcare'
            }

        self.keyword_impact_matrix = {
            'war': {
                'defense_stocks': +0.7,
                'tech_stocks': -0.5,
                'energy_stocks': +0.3
            },
            'pandemic': {
                'healthcare_stocks': +0.6,
                'tech_stocks': +0.4,
                'retail_stocks': -0.3
            }
        }

    def extract_sentiment_features(self, headline):
        """
        Extract nuanced sentiment features from a headline.
        """
        features = {
            'raw_sentiment': self._calculate_raw_sentiment(headline),
            'sector_impact': self._determine_sector_impact(headline),
            'keyword_weights': self._extract_keyword_impacts(headline)
        }
        return features

    def _calculate_raw_sentiment(self, headline):
        """
        Uses the VADER compound score from polarity_analysis for a numerical feature.
        """
        return get_compound_score(headline)

    def _determine_sector_impact(self, headline):
        """
        Determine how the headline impacts different stock sectors based on keywords.
        """
        sector_impacts = {}
        for sector, keywords in self.sector_sentiment_map.items():
            for keyword, sentiment in keywords.items():
                if keyword.lower() in headline.lower():
                    sector_impacts[sector] = sentiment
        return sector_impacts

    def _extract_keyword_impacts(self, headline):
        """
        Extract specific keyword impacts on stock performance using a predefined matrix.
        """
        keyword_weights = {}
        for keyword, sector_impacts in self.keyword_impact_matrix.items():
            if keyword.lower() in headline.lower():
                keyword_weights[keyword] = sector_impacts
        return keyword_weights