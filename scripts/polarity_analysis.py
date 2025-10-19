from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def get_sentiment_score(text: str) -> str:
    """
    Returns a categorical sentiment label (Positive, Neutral, Negative).
    """
    if not text or not isinstance(text, str):
        return "Neutral"

    score = _analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    return "Neutral"


def get_compound_score(text: str) -> float:
    """
    Returns the compound sentiment score (-1 to 1).
    """
    if not text or not isinstance(text, str):
        return 0.0
    return _analyzer.polarity_scores(text)["compound"]
