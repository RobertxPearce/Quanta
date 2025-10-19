from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Creates an analyzer object to reuse
analyzer = SentimentIntensityAnalyzer()


def get_sentiment_score(text):
    """
    Analyzes text and returns a 'compound' sentiment score.
    - Postive: > 0.05
    - Neutral: -0.05 to 0.05
    - Negative: < -0.05
    """
    # .polarity_scores() returns a dictionary: { 'neg', 'neu', 'pos', 'compound' }
    # We only care about the 'compound' score, which is a single float.
    # compound score is a numerical value that summarizes the overall emotional tone
    # compound = (sum of valence scores) / sqrt((sum of valence scores^2) + 15)

    score = analyzer.polarity_scores(text)['compound']

    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'


def get_compound_score(text):
    """
    Analyzes text and returns the raw compound sentiment score (float).
    """
    return analyzer.polarity_scores(text)['compound']