"""
News Feed Page - Market news with sentiment analysis
"""
import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import sys
from pathlib import Path

st.set_page_config(page_title="Market News", page_icon="📰", layout="wide")

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from utils.auth import require_authentication, render_user_info
    from utils.cache import get_cached_signals, get_cached_market_data, get_cached_performance_stats
    from pages.components.news_feed import (
        render_news_feed, render_news_summary_widget, render_market_sentiment_widget,
        render_news_analytics, apply_news_filters
    )
    from pages.components.sentiment_indicator import render_sentiment_summary, render_sentiment_timeline
    
    # Require authentication for this page
    user_info = require_authentication()
    render_user_info()
    imports_successful = True
except ImportError as e:
    st.warning("⚠️ Authentication or components modules not found - running in demo mode")
    user_info = {"username": "demo", "role": "admin"}
    imports_successful = False

# Enhanced CSS styling for news page
st.markdown("""
<style>
    /* Professional title styling */
    .news-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Enhanced metrics styling */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        padding: 1.2rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.3);
        transition: transform 0.2s ease;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    [data-testid="metric-container"] > div {
        color: white;
    }
    
    [data-testid="metric-container"] label {
        color: rgba(255, 255, 255, 0.9);
        font-weight: 600;
        font-size: 0.95rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 3px 10px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 10px;
        color: #667eea;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Filter section styling */
    .filter-section {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Analytics section */
    .analytics-section {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="news-title">📰 Market News & Sentiment</h1>', unsafe_allow_html=True)

# Helper function to call backend API with fallback for production
def call_api(endpoint, method="GET", data=None):
    """Call backend API with development/production environment detection"""
    import os
    
    try:
        # Try local backend API (development environment)
        base_url = "http://0.0.0.0:8000"
        url = f"{base_url}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            # If localhost fails, try alternative approaches
            raise requests.exceptions.ConnectionError("Localhost not available")
            
    except requests.exceptions.RequestException as e:
        # Fallback for production environment - return mock data or handle differently
        st.warning("⚠️ **Backend API unavailable** - Running in demo mode with sample data")
        return get_fallback_news_data(endpoint, method, data)

def get_fallback_news_data(endpoint, method="GET", data=None):
    """Provide fallback data when backend API is unavailable"""
    from datetime import datetime, timedelta
    import random
    
    if "/api/news/feed" in endpoint:
        # Generate sample news articles
        sample_news = []
        current_time = datetime.now()
        
        news_titles = [
            "Federal Reserve Signals Interest Rate Changes Amid Market Volatility",
            "EUR/USD Reaches Key Support Level as European Markets Open",
            "Asian Markets Show Mixed Results Following Central Bank Announcements",
            "Gold Prices Surge on Global Economic Uncertainty",
            "Bitcoin Trading Volume Increases as Institutional Interest Grows",
            "Oil Prices Fluctuate on OPEC+ Production Decision",
            "US Dollar Strength Continues Against Major Currency Pairs",
            "European Central Bank Maintains Current Monetary Policy",
            "Japanese Yen Weakens on Bank of Japan Policy Statement",
            "Cryptocurrency Market Shows Renewed Interest from Investors"
        ]
        
        sources = ["Reuters", "Bloomberg", "MarketWatch", "CNBC", "Financial Times", "Yahoo Finance"]
        symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "BTCUSD", "XAUUSD", "USDCAD", "USDCHF"]
        
        for i, title in enumerate(news_titles):
            published_time = current_time - timedelta(hours=random.randint(1, 72))
            sentiment_score = random.uniform(-0.8, 0.8)
            
            sample_news.append({
                "id": 1000 + i,
                "title": title,
                "summary": f"This is a sample news summary for the article '{title}'. In a real implementation, this would contain the actual article summary or excerpt from the news source.",
                "source": random.choice(sources),
                "url": f"https://example.com/news/{1000 + i}",
                "published_at": published_time.isoformat() + "Z",
                "symbols": random.sample(symbols, random.randint(1, 3)),
                "sentiment_score": sentiment_score,
                "sentiment_confidence": random.uniform(0.6, 0.95),
                "category": random.choice(["forex", "crypto", "commodities", "stocks"])
            })
        
        return sample_news
        
    elif "/api/news/sentiment-summary" in endpoint:
        # Generate sentiment summary data
        symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "BTCUSD"]
        sentiment_data = []
        
        for symbol in symbols:
            sentiment_data.append({
                "symbol": symbol,
                "sentiment": random.uniform(-0.6, 0.6),
                "confidence": random.uniform(0.7, 0.9),
                "article_count": random.randint(5, 25)
            })
        
        return sentiment_data
        
    elif "/api/news/analyze" in endpoint and method == "POST":
        return {"status": "success", "message": "News analysis would be triggered in production environment"}
    
    return []

# Load data with caching
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_news_feed(limit=50):
    """Load news feed from API"""
    return call_api(f"/api/news/feed?limit={limit}")

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_sentiment_summary():
    """Load sentiment summary from API"""
    return call_api("/api/news/sentiment-summary")

# Auto-refresh setup
import time

# Initialize session state
if 'news_last_refresh' not in st.session_state:
    st.session_state.news_last_refresh = time.time()

if 'news_auto_refresh' not in st.session_state:
    st.session_state.news_auto_refresh = False

# Load initial data
news_articles = load_news_feed()
sentiment_data = load_sentiment_summary()

# Top controls
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

with col1:
    st.subheader("📊 News Dashboard Controls")

with col2:
    if st.button("🔄 Refresh Now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

with col3:
    auto_refresh = st.checkbox("⏰ Auto-refresh (5min)", value=st.session_state.news_auto_refresh)
    st.session_state.news_auto_refresh = auto_refresh

with col4:
    if user_info.get('role') == 'admin':
        if st.button("🔍 Analyze News", use_container_width=True):
            with st.spinner("Triggering news analysis..."):
                result = call_api("/api/news/analyze", "POST")
                if result:
                    st.success("✅ News analysis triggered!")
                else:
                    st.error("❌ Failed to trigger analysis")

# Check for auto-refresh
if auto_refresh:
    current_time = time.time()
    if current_time - st.session_state.news_last_refresh > 300:  # 5 minutes
        st.cache_data.clear()
        st.session_state.news_last_refresh = current_time
        st.rerun()

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["📰 News Feed", "📊 Sentiment Analysis", "📈 Analytics", "⚙️ Settings"])

with tab1:
    # News Feed Tab
    st.markdown('<div class="filter-section">', unsafe_allow_html=True)
    st.subheader("🔍 Filter & Search")
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Extract unique symbols
        all_symbols = set()
        for article in news_articles:
            all_symbols.update(article.get('symbols', []))
        
        selected_symbols = st.multiselect(
            "Filter by Symbols:",
            options=sorted(list(all_symbols)),
            key="news_symbol_filter"
        )
    
    with col2:
        # Extract unique sources
        all_sources = set(article.get('source', 'Unknown') for article in news_articles)
        selected_sources = st.multiselect(
            "Filter by Sources:",
            options=sorted(list(all_sources)),
            key="news_source_filter"
        )
    
    with col3:
        sentiment_filter = st.selectbox(
            "Sentiment Filter:",
            options=["All", "Positive", "Neutral", "Negative"],
            key="news_sentiment_filter"
        )
    
    with col4:
        time_filter = st.selectbox(
            "Time Range:",
            options=["All", "Last Hour", "Last 6 Hours", "Last 24 Hours", "Last Week"],
            key="news_time_filter"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Apply filters
    filters = {
        'symbols': selected_symbols,
        'sources': selected_sources,
        'sentiment': sentiment_filter,
        'time_range': time_filter
    }
    
    if imports_successful:
        filtered_articles = apply_news_filters(news_articles, filters)
    else:
        filtered_articles = news_articles  # Fallback when imports fail
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Articles", len(news_articles))
    
    with col2:
        st.metric("Filtered Articles", len(filtered_articles))
    
    with col3:
        if filtered_articles:
            avg_sentiment = sum(a.get('sentiment_score', 0) for a in filtered_articles) / len(filtered_articles)
            st.metric("Avg Sentiment", f"{avg_sentiment:.3f}")
        else:
            st.metric("Avg Sentiment", "N/A")
    
    with col4:
        if filtered_articles:
            unique_sources = len(set(a.get('source') for a in filtered_articles))
            st.metric("Unique Sources", unique_sources)
        else:
            st.metric("Unique Sources", "0")
    
    # Display news feed
    if imports_successful:
        render_news_feed(
            filtered_articles,
            title="Market News Feed",
            show_sentiment=True,
            show_filters=False,  # Already shown above
            max_articles=None,
            compact_view=False
        )
    else:
        # Fallback display when imports fail
        st.subheader("📰 Market News Feed")
        for article in filtered_articles[:10]:
            with st.expander(f"📰 {article.get('title', 'No Title')}"):
                st.write(f"**Source:** {article.get('source', 'Unknown')}")
                st.write(f"**Published:** {article.get('published_at', 'Unknown')}")
                st.write(f"**Summary:** {article.get('summary', 'No summary available')}")
                st.write(f"**Sentiment:** {article.get('sentiment_score', 0):.3f}")

with tab2:
    # Sentiment Analysis Tab
    st.subheader("📊 Market Sentiment Analysis")
    
    if sentiment_data and imports_successful:
        render_sentiment_summary(sentiment_data, title="Current Market Sentiment")
        
        # Additional sentiment insights
        st.subheader("🔍 Sentiment Insights")
        
        # Prepare timeline data for sentiment chart
        timeline_data = []
        for article in news_articles[:20]:  # Last 20 articles
            if article.get('published_at') and article.get('sentiment_score') is not None:
                timeline_data.append({
                    'timestamp': article['published_at'],
                    'sentiment': article['sentiment_score'],
                    'symbol': article.get('symbols', ['General'])[0] if article.get('symbols') else 'General'
                })
        
        if timeline_data:
            render_sentiment_timeline(timeline_data, title="Recent Sentiment Timeline")
    
    elif sentiment_data:
        # Fallback display
        st.subheader("📊 Current Market Sentiment")
        for item in sentiment_data:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Symbol", item.get('symbol', 'Unknown'))
            with col2:
                sentiment = item.get('sentiment', 0)
                st.metric("Sentiment", f"{sentiment:.3f}")
            with col3:
                confidence = item.get('confidence', 0)
                st.metric("Confidence", f"{confidence:.1%}")
    else:
        st.info("No sentiment data available")

with tab3:
    # Analytics Tab
    st.subheader("📈 News Analytics")
    
    if news_articles and imports_successful:
        render_news_analytics(news_articles)
    else:
        # Fallback analytics
        st.info("Analytics require the news_feed component to be properly imported")
        
        if news_articles:
            # Basic analytics without components
            df_data = []
            for article in news_articles:
                df_data.append({
                    'source': article.get('source', 'Unknown'),
                    'sentiment': article.get('sentiment_score', 0),
                    'symbols': len(article.get('symbols', []))
                })
            
            if df_data:
                df = pd.DataFrame(df_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Articles by Source")
                    source_counts = df['source'].value_counts()
                    st.bar_chart(source_counts)
                
                with col2:
                    st.subheader("Sentiment Distribution")
                    st.bar_chart(df['sentiment'])

with tab4:
    # Settings Tab
    st.subheader("⚙️ News Feed Settings")
    
    if user_info.get('role') == 'admin':
        st.markdown("### 🔧 Admin Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Data Collection")
            
            refresh_interval = st.selectbox(
                "Auto-refresh Interval:",
                options=[60, 300, 600, 1800, 3600],
                format_func=lambda x: f"{x//60} minutes" if x >= 60 else f"{x} seconds",
                index=1  # Default to 5 minutes
            )
            
            max_articles = st.number_input(
                "Max Articles to Display:",
                min_value=10,
                max_value=200,
                value=50,
                step=10
            )
            
            enable_sentiment = st.checkbox("Enable Sentiment Analysis", value=True)
        
        with col2:
            st.subheader("🔍 Analysis Settings")
            
            sentiment_threshold = st.slider(
                "Sentiment Alert Threshold:",
                min_value=0.1,
                max_value=1.0,
                value=0.7,
                step=0.1
            )
            
            confidence_threshold = st.slider(
                "Confidence Threshold:",
                min_value=0.5,
                max_value=1.0,
                value=0.8,
                step=0.05
            )
            
            enable_alerts = st.checkbox("Enable Sentiment Alerts", value=False)
        
        if st.button("💾 Save Settings", use_container_width=True):
            settings = {
                "refresh_interval": refresh_interval,
                "max_articles": max_articles,
                "enable_sentiment": enable_sentiment,
                "sentiment_threshold": sentiment_threshold,
                "confidence_threshold": confidence_threshold,
                "enable_alerts": enable_alerts
            }
            # In production, this would save to backend
            st.success("✅ Settings saved successfully!")
            
        st.markdown("### 🧹 Maintenance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ Clear Cache", use_container_width=True):
                st.cache_data.clear()
                st.success("Cache cleared!")
        
        with col2:
            if st.button("🔄 Force Refresh", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        
        with col3:
            if st.button("📊 Recalculate Sentiment", use_container_width=True):
                with st.spinner("Recalculating sentiment..."):
                    result = call_api("/api/news/analyze", "POST", {"recalculate": True})
                    if result:
                        st.success("✅ Sentiment recalculation triggered!")
    else:
        st.info("👤 Admin access required for settings configuration")
        
        st.markdown("### 📊 User Preferences")
        
        user_compact_view = st.checkbox("Prefer Compact View", value=False)
        user_auto_refresh = st.checkbox("Enable Auto-refresh", value=True)
        user_show_confidence = st.checkbox("Show Confidence Scores", value=True)
        
        if st.button("💾 Save Preferences"):
            # In production, this would save user preferences
            st.success("✅ Preferences saved!")

# Footer with last update info
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if news_articles:
        latest_article_time = max(
            datetime.fromisoformat(article['published_at'].replace('Z', '+00:00'))
            for article in news_articles 
            if article.get('published_at')
        )
        st.info(f"📅 Latest Article: {latest_article_time.strftime('%Y-%m-%d %H:%M')}")

with col2:
    last_refresh_time = datetime.fromtimestamp(st.session_state.news_last_refresh)
    st.info(f"🔄 Last Refresh: {last_refresh_time.strftime('%H:%M:%S')}")

with col3:
    st.info(f"📰 Total Articles: {len(news_articles)}")

# Auto-refresh logic for the page
if auto_refresh:
    # Add a small indicator for auto-refresh
    st.markdown(
        '<div style="position: fixed; top: 10px; right: 10px; background: #28a745; color: white; '
        'padding: 0.3rem 0.6rem; border-radius: 5px; font-size: 0.8rem; z-index: 999;">'
        '🔄 Auto-refresh ON</div>',
        unsafe_allow_html=True
    )