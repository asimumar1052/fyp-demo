"""Defines the TweetResponse (and Author) model used for validating tweet data."""

from typing import Optional, Dict
from pydantic import BaseModel

class Author(BaseModel):
    """Model for author of a tweet."""
    name: Optional[str] = None
    screen_name: Optional[str] = None
    image: Optional[str] = None
    blue_verified: Optional[bool] = None

class TweetResponse(BaseModel):
    """Model for tweet response data."""
    id: Optional[str] = None
    created_at: Optional[str] = None
    text: Optional[str] = None
    lang: Optional[str] = None
    likes: Optional[int] = None
    retweets: Optional[int] = None
    bookmarks: Optional[int] = None
    quotes: Optional[int] = None
    replies: Optional[int] = None
    author: Optional[Author] = None
    media: Optional[Dict] = None
    sentiment: Optional[Dict[str, float | str]] = None
    fact_check_trigger: Optional[Dict[str, float | str]] = None
    fake_news_detection: Optional[Dict[str, float | str]] = None
