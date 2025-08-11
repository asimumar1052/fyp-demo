"""Provides sentiment analysis for text using Hugging Face Inference API."""

import requests
import os
from typing import Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Hugging Face Inference API configuration
API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment"

def query_huggingface_api(text: str) -> Dict:
    """Query Hugging Face Inference API for sentiment analysis."""
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
                    raise Exception("Model is loading, please try again in a few seconds")
                else:
                    raise Exception(f"API error: {result['error']}")
            
            return result
            
        elif response.status_code == 429:
            raise Exception("Rate limit exceeded. Please try again later or upgrade your plan.")
        elif response.status_code == 401:
            raise Exception("Invalid API token. Please check your HUGGINGFACE_API_TOKEN.")
        else:
            raise Exception(f"Hugging Face API error: {response.status_code} - {response.text}")
            
    except requests.exceptions.Timeout:
        raise Exception("API request timed out. Please try again.")
    except requests.exceptions.ConnectionError:
        raise Exception("Unable to connect to Hugging Face API. Check your internet connection.")

def analyze_sentiment(text: str) -> dict:
    """Returns sentiment label and confidence score using Hugging Face Inference API."""
    try:
        # Validate input
        if not text or not text.strip():
            return {
                "label": "Neutral",
                "confidence": 0.0
            }
        
        # Preprocess text (username and link placeholders as expected by the model)
        processed_text = preprocess(text)
        
        # Query the API
        result = query_huggingface_api(processed_text)
        
        # Parse the API response
        if isinstance(result, list) and len(result) > 0:
            # Get the prediction with highest confidence
            predictions = result[0]
            if isinstance(predictions, list) and len(predictions) > 0:
                top_prediction = max(predictions, key=lambda x: x.get('score', 0))
                
                # Map labels to consistent format
                label_mapping = {
                    'LABEL_0': 'Negative',  # NEGATIVE
                    'LABEL_1': 'Neutral',   # NEUTRAL  
                    'LABEL_2': 'Positive',  # POSITIVE
                }
                
                raw_label = top_prediction.get('label', 'LABEL_1')
                confidence = top_prediction.get('score', 0.0)
                
                return {
                    "label": label_mapping.get(raw_label, 'Neutral'),
                    "confidence": confidence
                }
            else:
                # Fallback if predictions format is unexpected
                return {
                    "label": "Neutral",
                    "confidence": 0.0
                }
        else:
            # Fallback if API response format is unexpected
            return {
                "label": "Neutral",
                "confidence": 0.0
            }
            
    except Exception as e:
        print(f"Error in sentiment analysis: {str(e)}")
        # Fallback to neutral sentiment if API fails
        return {
            "label": "Neutral", 
            "confidence": 0.0
        }

def preprocess(text):
    """Preprocess text for the Twitter RoBERTa model."""
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)