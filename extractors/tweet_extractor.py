"""Extracts tweet data from a given tweet URL using the RapidAPI service."""

import re
import os
import httpx
from dotenv import load_dotenv

load_dotenv()

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
RAPIDAPI_HOST = os.getenv("RAPIDAPI_HOST")

if RAPIDAPI_KEY is None or RAPIDAPI_HOST is None:
    raise RuntimeError("RAPIDAPI_KEY and RAPIDAPI_HOST must be set in environment variables.")

def extract_tweet_id(tweet_url: str) -> str:
    """Extracts the tweet ID from a tweet URL."""
    match = re.search(r"(twitter\.com|x\.com)/\w+/status/(\d+)", tweet_url)
    if not match:
        raise ValueError("Invalid tweet URL")
    return match.group(2)

async def extract_tweet_data(tweet_url: str) -> dict:
    """Extracts tweet data from the given tweet URL using RapidAPI."""
    tweet_id = extract_tweet_id(tweet_url)
    url = f"https://{RAPIDAPI_HOST}/tweet.php?id={tweet_id}"
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers) # type: ignore

    if response.status_code != 200:
        raise httpx.HTTPStatusError(
            f"Failed to fetch tweet data: {response.text}",
            request=response.request,
            response=response
        )

    data = response.json()
    result = {
        "id": data.get("id"),
        "created_at": data.get("created_at"),
        "text": data.get("text"),
        "lang": data.get("lang"),
        "likes": data.get("likes"),
        "retweets": data.get("retweets"),
        "bookmarks": data.get("bookmarks"),
        "quotes": data.get("quotes"),
        "replies": data.get("replies"),
        "author": {
            "name": data.get("author", {}).get("name"),
            "screen_name": data.get("author", {}).get("screen_name"),
            "image": data.get("author", {}).get("image"),
            "blue_verified": data.get("author", {}).get("blue_verified"),
        },
        "media": data.get("media"),
    }
    return result
