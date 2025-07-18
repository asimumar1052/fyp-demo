"""Main application file for the FastAPI server."""

from fastapi import FastAPI, HTTPException

from extractors.tweet_extractor import extract_tweet_data
from models.tweet_url import TweetURL
from models.tweet_response import TweetResponse
from analyzers.sentiment_analyzer import analyze_sentiment

app = FastAPI()

@app.post("/analyze-tweet", response_model=TweetResponse)
async def analyze_tweet(data: TweetURL):
    """Endpoint to analyze a tweet given its URL."""
    try:
        tweet = await extract_tweet_data(data.url)

        sentiment = analyze_sentiment(tweet['text'])
        tweet['sentiment'] = sentiment
        
        return tweet
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid tweet URL format") from exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
