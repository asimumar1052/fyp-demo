"""Provides fake news detection using Hugging Face Inference API."""

import requests
import os
from typing import Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Hugging Face Inference API configuration for fake news detection
API_URL = "https://api-inference.huggingface.co/models/mrm8488/bert-tiny-finetuned-fake-news-detection"

def query_fake_news_api(text: str) -> Dict:
    """Query Hugging Face Inference API for fake news detection."""
    # Get API token (load fresh each time to ensure it's available)
    api_token = os.getenv("HUGGINGFACE_API_TOKEN")
    
    # Check if API token is available
    if not api_token:
        raise Exception("HUGGINGFACE_API_TOKEN not found in environment variables")
    
    headers = {"Authorization": f"Bearer {api_token}"}
    payload = {"inputs": text}
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            # Handle model loading state
            if isinstance(result, dict) and "error" in result:
                if "loading" in result["error"].lower():
                    raise Exception("Fake news detection model is loading, please try again in a few seconds")
                else:
                    raise Exception(f"Fake news API error: {result['error']}")
            
            return result
            
        elif response.status_code == 429:
            raise Exception("Rate limit exceeded for fake news detection API. Please try again later.")
        elif response.status_code == 401:
            raise Exception("Invalid API token for fake news detection.")
        else:
            raise Exception(f"Fake news API error: {response.status_code} - {response.text}")
            
    except requests.exceptions.Timeout:
        raise Exception("Fake news detection API request timed out. Please try again.")
    except requests.exceptions.ConnectionError:
        raise Exception("Unable to connect to fake news detection API. Check your internet connection.")

def detect_fake_news(text: str) -> dict:
    """Returns fake news detection results (FAKE/REAL) with confidence score."""
    try:
        # Validate input
        if not text or not text.strip():
            return {
                "label": "REAL",
                "confidence": 0.0
            }
        
        # Query the API
        result = query_fake_news_api(text)
        
        # Parse the API response
        if isinstance(result, list) and len(result) > 0:
            # Get the prediction with highest confidence
            predictions = result[0]
            if isinstance(predictions, list) and len(predictions) > 0:
                top_prediction = max(predictions, key=lambda x: x.get('score', 0))
                
                # Extract label and confidence
                raw_label = top_prediction.get('label', 'REAL')
                confidence = top_prediction.get('score', 0.0)
                
                # Normalize label format
                normalized_label = raw_label.upper()
                if normalized_label not in ['FAKE', 'REAL']:
                    # Map common variations
                    if 'fake' in raw_label.lower() or 'false' in raw_label.lower():
                        normalized_label = 'FAKE'
                    else:
                        normalized_label = 'REAL'
                
                return {
                    "label": normalized_label,
                    "confidence": float(confidence)
                }
            else:
                # Fallback if predictions format is unexpected
                return {
                    "label": "REAL",
                    "confidence": 0.0
                }
        else:
            # Fallback if API response format is unexpected
            return {
                "label": "REAL",
                "confidence": 0.0
            }
            
    except Exception as e:
        print(f"Error in fake news detection: {str(e)}")
        # Fallback to REAL if API fails
        return {
            "label": "REAL", 
            "confidence": 0.0
        }
