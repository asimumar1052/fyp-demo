"""Provides sentiment analysis for text using VADER."""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text: str) -> dict:
    """Returns sentiment label and confidence score."""
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']

    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    confidence = round(abs(compound), 2)

    return {
        "label": label,
        "confidence": confidence
    }