"""Defines the TweetURL model used for validating tweet URLs."""

from pydantic import BaseModel

class TweetURL(BaseModel):
    """Model for validating tweet URLs."""
    url: str
