from typing import Optional, Dict, Any, List
from .polarity_analysis import get_compound_score
from .get_data import get_ticker_sectors


class StockSentimentMapper:
    """
    Maps stock tickers to their sectors and computes sentiment features
    based on sector context and keyword analysis.
    """

    def __init__(self, tickers_to_analyze: Optional[List[str]] = None):
        # Define general sentiment tendencies per sector
        self.sector_sentiment_map = {
            "defense": {"war": "positive", "conflict": "positive"},
            "tech": {"war": "negative", "chip_shortage": "negative"},
            "healthcare": {"pandemic": "complex", "medical_breakthrough": "positive"},
        }

        # Predefined keyword-level weights for sector effects
        self.keyword_impact_matrix = {
            "war": {"defense_stocks": +0.7, "tech_stocks": -0.5},
            "pandemic": {"healthcare_stocks": +0.6, "tech_stocks": +0.4},
        }

        # Map tickers to sectors — API or fallback
        if tickers_to_analyze:
            self.stock_sector_mapping = get_ticker_sectors(tickers_to_analyze)
        else:
            self.stock_sector_mapping = {
                "LMT": "defense",
                "MSFT": "tech",
                "GOOGL": "tech",
                "BA": "defense",
                "NVDA": "tech",
                "AAPL": "tech",
                "JNJ": "healthcare",
            }

    def extract_sentiment_features(self, headline: str) -> Dict[str, Any]:
        """
        Extracts sentiment-related features from a news headline.
        Returns:
            dict: containing raw sentiment, sector impact, and keyword weights.
        """
        return {
            "raw_sentiment": self._calculate_raw_sentiment(headline),
            "sector_impact": self._determine_sector_impact(headline),
            "keyword_weights": self._extract_keyword_impacts(headline),
        }

    def _calculate_raw_sentiment(self, headline: str) -> float:
        """Numerical sentiment score via VADER compound score."""
        return get_compound_score(headline)

    def _determine_sector_impact(self, headline: str) -> Dict[str, str]:
        """Identify sector-specific impact words present in a headline."""
        return {
            sector: sentiment
            for sector, keywords in self.sector_sentiment_map.items()
            for keyword, sentiment in keywords.items()
            if keyword.lower() in headline.lower()
        }

    def _extract_keyword_impacts(self, headline: str) -> Dict[str, Dict[str, float]]:
        """Map relevant keywords to their predefined sectoral weight impacts."""
        return {
            keyword: impact
            for keyword, impact in self.keyword_impact_matrix.items()
            if keyword.lower() in headline.lower()
        }
