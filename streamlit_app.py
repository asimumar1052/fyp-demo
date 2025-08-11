"""
Streamlit Frontend for Tweelyzer - Tweet Analysis Tool
"""

import streamlit as st
import requests
import json
from datetime import datetime

# Configure the page
st.set_page_config(
    page_title="🐦 Tweelyzer - Tweet Analyzer",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #1DA1F2, #14171A);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .tweet-container {
        border: 1px solid #e1e8ed;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        background-color: #f8f9fa;
    }
    .sentiment-positive {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
    .sentiment-negative {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
    }
    .sentiment-neutral {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🐦 Tweelyzer</h1>
    <p>Advanced Tweet Sentiment Analysis with AI</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    backend_url = st.text_input(
        "Backend URL",
        value="http://127.0.0.1:8000",
        help="URL of your FastAPI backend server"
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This tool analyzes tweets using advanced AI models:
    - **CardiffNLP Twitter RoBERTa** via Hugging Face API
    - **BART Large MNLI** for fact-check trigger detection
    - **BERT Tiny** for fake news detection
    - **Real-time tweet extraction**
    - **Beautiful data visualization**
    - **Cloud-powered inference** (no local GPU needed!)
    """)

# Main input section
st.header("📝 Enter Tweet URL")
col1, col2 = st.columns([3, 1])

with col1:
    tweet_url = st.text_input(
        "Tweet URL",
        placeholder="https://twitter.com/username/status/123456789...",
        help="Paste the full URL of the tweet you want to analyze",
        label_visibility="hidden"
    )

with col2:
    analyze_button = st.button("🔍 Analyze Tweet", type="primary", use_container_width=True)

# Analysis section
if analyze_button and tweet_url:
    if not tweet_url.strip():
        st.error("❌ Please enter a valid tweet URL")
    else:
        with st.spinner("🔄 Analyzing tweet... This may take a moment for the first analysis."):
            try:
                # Make API request
                response = requests.post(
                    f"{backend_url}/analyze-tweet",
                    json={"url": tweet_url},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Success message
                    st.success("✅ Analysis completed successfully!")
                    
                    # Tweet Content Section
                    st.markdown("---")
                    st.header("📄 Tweet Content")
                    
                    with st.container():
                        st.markdown('<div class="tweet-container">', unsafe_allow_html=True)
                        
                        # Tweet text
                        tweet_text = data.get('text', 'N/A')
                        st.markdown(f"**Tweet Text:**")
                        st.markdown(f"> {tweet_text}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Sentiment Analysis Section
                    st.markdown("---")
                    st.header("🎭 Sentiment Analysis")
                    
                    sentiment_info = data.get('sentiment', {})
                    if isinstance(sentiment_info, dict):
                        sentiment_label = sentiment_info.get('label', 'Unknown')
                        sentiment_confidence = sentiment_info.get('confidence', 0)
                    else:
                        sentiment_label = str(sentiment_info)
                        sentiment_confidence = 0
                    
                    # Sentiment display with color coding
                    sentiment_lower = sentiment_label.lower()
                    if sentiment_lower == 'positive':
                        sentiment_class = 'sentiment-positive'
                        emoji = '😊'
                    elif sentiment_lower == 'negative':
                        sentiment_class = 'sentiment-negative'
                        emoji = '😔'
                    else:
                        sentiment_class = 'sentiment-neutral'
                        emoji = '😐'
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="{sentiment_class}">
                            <h3>{emoji} {sentiment_label}</h3>
                            <p>Sentiment Classification</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        if sentiment_confidence > 0:
                            st.metric(
                                "Confidence Score",
                                f"{sentiment_confidence:.1%}",
                                help="How confident the AI model is about this prediction"
                            )
                        else:
                            st.metric("Confidence Score", "N/A")
                    
                    with col3:
                        # Model information
                        st.info("**Model Used:**\nCardiffNLP Twitter RoBERTa\n(via Hugging Face API)")
                    
                    # Fact Check Analysis Section
                    st.markdown("---")
                    st.header("🔍 Fact Check Analysis")
                    
                    fact_check_info = data.get('fact_check_trigger', {})
                    if isinstance(fact_check_info, dict):
                        fact_check_label = fact_check_info.get('label', 'Unknown')
                        fact_check_confidence = fact_check_info.get('confidence', 0)
                    else:
                        fact_check_label = str(fact_check_info)
                        fact_check_confidence = 0
                    
                    # Fact check display with color coding
                    needs_fact_check = fact_check_label == 'Needs fact check'
                    if needs_fact_check:
                        fact_check_class = 'sentiment-negative'
                        fact_check_emoji = '⚠️'
                    else:
                        fact_check_class = 'sentiment-positive'
                        fact_check_emoji = '✅'
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="{fact_check_class}">
                            <h3>{fact_check_emoji} {fact_check_label}</h3>
                            <p>Fact Check Trigger</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        if fact_check_confidence > 0:
                            st.metric(
                                "Confidence Score",
                                f"{fact_check_confidence:.1%}",
                                help="How confident the AI model is about this prediction"
                            )
                        else:
                            st.metric("Confidence Score", "N/A")
                    
                    with col3:
                        st.info("**Model Used:**\nFacebook BART Large MNLI\n(Zero-shot Classification)")
                    
                    # Fake News Detection Section (only if fact check is needed)
                    if needs_fact_check:
                        st.markdown("---")
                        st.header("🚨 Fake News Detection")
                        
                        fake_news_info = data.get('fake_news_detection', {})
                        if isinstance(fake_news_info, dict):
                            fake_news_label = fake_news_info.get('label', 'Unknown')
                            fake_news_confidence = fake_news_info.get('confidence', 0)
                        else:
                            fake_news_label = str(fake_news_info)
                            fake_news_confidence = 0
                        
                        # Fake news display with color coding
                        is_fake = fake_news_label.upper() == 'FAKE'
                        if is_fake:
                            fake_news_class = 'sentiment-negative'
                            fake_news_emoji = '🚨'
                        else:
                            fake_news_class = 'sentiment-positive'
                            fake_news_emoji = '✅'
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="{fake_news_class}">
                                <h3>{fake_news_emoji} {fake_news_label}</h3>
                                <p>News Authenticity</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            if fake_news_confidence > 0:
                                st.metric(
                                    "Confidence Score",
                                    f"{fake_news_confidence:.1%}",
                                    help="How confident the AI model is about this prediction"
                                )
                            else:
                                st.metric("Confidence Score", "N/A")
                        
                        with col3:
                            st.info("**Model Used:**\nBERT Tiny Fine-tuned\n(Fake News Detection)")
                    else:
                        st.markdown("---")
                        st.header("🚨 Fake News Detection")
                        st.info("⏭️ **Skipped** - No fact checking needed based on content analysis.")
                    
                    # Tweet Metadata Section
                    st.markdown("---")
                    st.header("📊 Tweet Details")
                    
                    # Author Information (if available)
                    if 'author' in data and isinstance(data['author'], dict):
                        author = data['author']
                        st.subheader("👤 Author Information")
                        
                        author_cols = st.columns([1, 3])
                        with author_cols[0]:
                            # Profile image
                            if author.get('image'):
                                st.image(author['image'], width=80)
                        
                        with author_cols[1]:
                            # Author details
                            if author.get('name'):
                                verified_badge = " ✅" if author.get('blue_verified') else ""
                                st.markdown(f"**{author['name']}{verified_badge}**")
                            
                            if author.get('screen_name'):
                                st.markdown(f"@{author['screen_name']}")
                    
                    # Tweet Metrics
                    st.subheader("� Tweet Metrics")
                    metric_cols = st.columns(4)
                    
                    with metric_cols[0]:
                        if 'date' in data:
                            st.metric("� Date", data['date'])
                        else:
                            st.metric("� Date", "N/A")
                    
                    with metric_cols[1]:
                        if 'likes' in data:
                            st.metric("❤️ Likes", f"{data['likes']:,}")
                        else:
                            st.metric("❤️ Likes", "N/A")
                    
                    with metric_cols[2]:
                        if 'retweets' in data:
                            st.metric("🔄 Retweets", f"{data['retweets']:,}")
                        else:
                            st.metric("🔄 Retweets", "N/A")
                    
                    with metric_cols[3]:
                        if 'replies' in data:
                            st.metric("� Replies", f"{data['replies']:,}")
                        else:
                            st.metric("� Replies", "N/A")
                    
                    # Raw Data Section (Expandable)
                    st.markdown("---")
                    with st.expander("🔍 View Raw API Response"):
                        st.json(data)
                    
                    # Download option
                    st.markdown("---")
                    st.header("💾 Export Results")
                    
                    # Prepare data for download
                    export_data = {
                        "tweet_url": tweet_url,
                        "analysis_timestamp": datetime.now().isoformat(),
                        "results": data
                    }
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.download_button(
                            label="📄 Download as JSON",
                            data=json.dumps(export_data, indent=2),
                            file_name=f"tweet_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    
                    with col2:
                        # Create a simple text report
                        # Handle author information properly
                        author_name = "N/A"
                        if 'author' in data and isinstance(data['author'], dict):
                            author_obj = data['author']
                            author_name = author_obj.get('name', author_obj.get('screen_name', 'N/A'))
                        
                        # Extract analysis results
                        fact_check_info = data.get('fact_check_trigger', {})
                        fact_check_label = fact_check_info.get('label', 'N/A')
                        fact_check_confidence = fact_check_info.get('confidence', 0)
                        
                        fake_news_info = data.get('fake_news_detection', {})
                        fake_news_label = fake_news_info.get('label', 'N/A')
                        fake_news_confidence = fake_news_info.get('confidence', 0)
                        
                        report = f"""
TWEET ANALYSIS REPORT
===================

Tweet URL: {tweet_url}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Tweet Content:
{tweet_text}

ANALYSIS RESULTS:
================

1. Sentiment Analysis:
   - Classification: {sentiment_label}
   - Confidence: {sentiment_confidence:.1%}

2. Fact Check Trigger:
   - Classification: {fact_check_label}
   - Confidence: {fact_check_confidence:.1%}

3. Fake News Detection:
   - Classification: {fake_news_label}
   - Confidence: {fake_news_confidence:.1%}

Tweet Metadata:
==============
- Author: {author_name}
- Date: {data.get('date', 'N/A')}
- Likes: {data.get('likes', 'N/A')}
- Retweets: {data.get('retweets', 'N/A')}

Generated by Tweelyzer - Advanced Tweet Analysis Tool
                        """
                        
                        st.download_button(
                            label="📝 Download as Text",
                            data=report.strip(),
                            file_name=f"tweet_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                
                elif response.status_code == 400:
                    error_detail = response.json().get('detail', 'Invalid tweet URL format')
                    st.error(f"❌ Error: {error_detail}")
                    st.info("💡 Please check that your tweet URL is valid and accessible.")
                
                elif response.status_code == 500:
                    error_detail = response.json().get('detail', 'Internal server error')
                    st.error(f"❌ Server Error: {error_detail}")
                    st.info("💡 There was an issue processing your request. Please try again.")
                
                else:
                    st.error(f"❌ Unexpected error: HTTP {response.status_code}")
                    st.error(f"Response: {response.text}")
            
            except requests.exceptions.Timeout:
                st.error("⏰ Request timed out. The server might be starting up or processing. Please try again.")
            
            except requests.exceptions.ConnectionError:
                st.error(f"🔌 Cannot connect to backend server at {backend_url}")
                st.info("💡 Make sure your FastAPI backend is running and accessible.")
            
            except requests.exceptions.RequestException as e:
                st.error(f"🌐 Network error: {str(e)}")
            
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    Made with ❤️ using Streamlit and FastAPI | 
    Powered by Hugging Face Inference API (Sentiment + Fact Check + Fake News Detection)
</div>
""", unsafe_allow_html=True)
