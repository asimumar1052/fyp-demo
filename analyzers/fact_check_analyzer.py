"""Provides fact check trigger analysis using Hugging Face Inference API."""

import requests
import os
from typing import Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Hugging Face Inference API configuration for zero-shot classification
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"

def query_fact_check_api(text: str) -> Dict:
    """Query Hugging Face Inference API for fact check classification."""
    # Get API token (load fresh each time to ensure it's available)
    api_token = os.getenv("HUGGINGFACE_API_TOKEN")
    
    # Check if API token is available
    if not api_token:
        raise Exception("HUGGINGFACE_API_TOKEN not found in environment variables")
    
    headers = {"Authorization": f"Bearer {api_token}"}
    payload = {
        "inputs": text,
        "parameters": {
            "candidate_labels": ["Needs fact check", "No fact check needed"]
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            
            # Handle model loading state
            if isinstance(result, dict) and "error" in result:
                if "loading" in result["error"].lower():
                    raise Exception("Fact check model is loading, please try again in a few seconds")
                else:
                    raise Exception(f"Fact check API error: {result['error']}")
            
            return result
            
        elif response.status_code == 429:
            raise Exception("Rate limit exceeded for fact check API. Please try again later.")
        elif response.status_code == 401:
            raise Exception("Invalid API token for fact check analysis.")
        elif response.status_code == 503:
            raise Exception("Fact check model is currently overloaded. Please try again later.")
        else:
            raise Exception(f"Fact check API error: {response.status_code} - {response.text}")
            
    except requests.exceptions.Timeout:
        raise Exception("Fact check API request timed out. Model may be overloaded.")
    except requests.exceptions.ConnectionError:
        raise Exception("Unable to connect to fact check API. Check your internet connection.")

def analyze_fact_check_trigger(text: str) -> dict:
    """Returns whether a tweet needs fact checking using zero-shot classification."""
    try:
        # Validate input
        if not text or not text.strip():
            return {
                "label": "No fact check needed",
                "confidence": 0.0
            }
        
        # Query the API
        result = query_fact_check_api(text)
        
        # Parse the API response
        if isinstance(result, dict) and "labels" in result and "scores" in result:
            labels = result["labels"]
            scores = result["scores"]
            
            if len(labels) > 0 and len(scores) > 0:
                # Get the top prediction
                top_label = labels[0]
                top_confidence = scores[0]
                
                return {
                    "label": top_label,
                    "confidence": float(top_confidence)
                }
            else:
                # Fallback if response format is unexpected
                return {
                    "label": "No fact check needed",
                    "confidence": 0.0
                }
        else:
            # Fallback if API response format is unexpected
            return {
                "label": "No fact check needed",
                "confidence": 0.0
            }
            
    except Exception as e:
        print(f"Error in fact check analysis: {str(e)}")
        
        # Smart fallback: Use keyword-based detection for potentially controversial content
        controversial_keywords = [
            'vaccine', 'covid', 'coronavirus', 'pandemic', 'government', 'conspiracy',
            'fake', 'hoax', 'secret', 'hidden', 'truth', 'lie', 'fraud', 'scam',
            'election', 'voting', 'rigged', 'stolen', 'climate change', 'global warming',
            'cure', 'medicine', 'drug', 'treatment', 'scientist', 'research', 'study',
            'flat earth', 'nasa', 'space', 'moon landing', '5g', 'microchip', 'tracking'
        ]
        
        text_lower = text.lower()
        for keyword in controversial_keywords:
            if keyword in text_lower:
                return {
                    "label": "Needs fact check",
                    "confidence": 0.7,
                    "note": "Triggered by keyword-based fallback due to API unavailability"
                }
        
        # If no controversial keywords found, assume no fact check needed
        return {
            "label": "No fact check needed", 
            "confidence": 0.6,
            "note": "Fallback analysis - API unavailable"
        }
