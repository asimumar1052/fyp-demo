"""Main application file for the FastAPI server."""

from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from extractors.tweet_extractor import extract_tweet_data
from models.tweet_url import TweetURL
from models.tweet_response import TweetResponse
from analyzers.sentiment_analyzer import analyze_sentiment
from analyzers.fact_check_analyzer import analyze_fact_check_trigger
from analyzers.fake_news_detector import detect_fake_news

# Load environment variables
load_dotenv()

app = FastAPI()

@app.post("/analyze-tweet", response_model=TweetResponse)
async def analyze_tweet(data: TweetURL):
    """Endpoint to analyze a tweet given its URL."""
    try:
        tweet = await extract_tweet_data(data.url)

        # Stage 1: Sentiment Analysis
        sentiment = analyze_sentiment(tweet['text'])
        tweet['sentiment'] = sentiment
        
        # Stage 2: Claim Detection Analysis
        fact_check_result = analyze_fact_check_trigger(tweet['text'])
        tweet['fact_check_trigger'] = fact_check_result
        
        # Stage 3: Fake News Detection (only if claims are detected)
        if fact_check_result.get('label') == 'Needs fact check':
            fake_news_result = detect_fake_news(tweet['text'])
            tweet['fake_news_detection'] = fake_news_result
        else:
            # If no claims detected, mark as REAL by default
            tweet['fake_news_detection'] = {
                "label": "REAL",
                "confidence": 0.0,
                "note": "Skipped - No claims detected"
            }
        
        return tweet
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid tweet URL format") from exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
