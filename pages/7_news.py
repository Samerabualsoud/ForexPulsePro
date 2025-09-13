"""
Market News Page - Simplified
"""
import streamlit as st
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
    # Require authentication for this page
    user_info = require_authentication()
    render_user_info()
    imports_successful = True
except ImportError:
    st.warning("⚠️ Authentication modules not found - running in demo mode")
    user_info = {"username": "demo", "role": "admin"}
    imports_successful = False

# Clean, simple CSS styling
st.markdown("""
<style>
    /* Simple title styling */
    .news-title {
        font-size: 2.5rem;
        font-weight: 600;
        text-align: center;
        color: #2c3e50;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 3px solid #3498db;
    }
    
    /* News article cards */
    .news-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: box-shadow 0.2s ease;
    }
    
    .news-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .news-title-text {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 0.5rem;
        line-height: 1.4;
    }
    
    .news-meta {
        color: #6c757d;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    
    .news-summary {
        color: #495057;
        line-height: 1.6;
        margin-bottom: 1rem;
    }
    
    /* Sentiment indicators */
    .sentiment-positive {
        background: #d4edda;
        color: #155724;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .sentiment-negative {
        background: #f8d7da;
        color: #721c24;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .sentiment-neutral {
        background: #e2e3e5;
        color: #383d41;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3498db;
    }
    
    /* Filter section */
    .filter-section {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="news-title">📰 Market News</h1>', unsafe_allow_html=True)

# Helper function for API calls
def call_api(endpoint, method="GET", data=None):
    """Call backend API with fallback to demo data"""
    try:
        base_url = "http://0.0.0.0:8000"
        url = f"{base_url}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise requests.exceptions.ConnectionError("API not available")
            
    except requests.exceptions.RequestException:
        st.info("🔄 Running in demo mode")
        return get_demo_news_data(endpoint)

def get_demo_news_data(endpoint):
    """Provide demo news data"""
    import random
    
    if "/api/news/feed" in endpoint:
        current_time = datetime.now()
        
        news_items = [
            {
                "title": "Federal Reserve Signals Interest Rate Policy Changes",
                "summary": "The Federal Reserve indicated potential adjustments to interest rates following recent economic data, impacting currency markets globally.",
                "source": "Reuters",
                "sentiment": 0.2
            },
            {
                "title": "EUR/USD Reaches Key Technical Support Level",
                "summary": "The Euro-Dollar pair touched significant support as European markets opened, with traders watching for potential reversal signals.",
                "source": "Bloomberg",
                "sentiment": -0.3
            },
            {
                "title": "Asian Markets Show Mixed Performance",
                "summary": "Asian stock markets displayed varied results following overnight developments in global trade negotiations and monetary policy.",
                "source": "MarketWatch",
                "sentiment": 0.1
            },
            {
                "title": "Gold Prices Rally on Economic Uncertainty",
                "summary": "Precious metals saw increased demand as investors sought safe-haven assets amid global economic uncertainty and market volatility.",
                "source": "CNBC",
                "sentiment": 0.4
            },
            {
                "title": "Oil Markets React to OPEC+ Production Decisions",
                "summary": "Crude oil prices fluctuated following the latest OPEC+ meeting, with production targets affecting global energy markets.",
                "source": "Yahoo Finance",
                "sentiment": -0.1
            },
            {
                "title": "US Dollar Strengthens Against Major Currencies",
                "summary": "The US Dollar index gained ground against major trading partners as economic indicators showed continued strength in the US economy.",
                "source": "Financial Times",
                "sentiment": 0.3
            }
        ]
        
        # Add realistic timestamps and random details
        for i, item in enumerate(news_items):
            item["id"] = 1000 + i
            item["published_at"] = (current_time - timedelta(hours=random.randint(1, 24))).isoformat() + "Z"
            item["symbols"] = random.sample(["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "USOIL"], random.randint(1, 2))
            
        return news_items
    
    elif "/api/news/sentiment" in endpoint:
        return {
            "overall_sentiment": random.uniform(-0.2, 0.3),
            "sentiment_confidence": random.uniform(0.7, 0.9),
            "trending_positive": ["GBPUSD", "XAUUSD"],
            "trending_negative": ["USDJPY"]
        }
    
    return []

# Load data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_news_data():
    """Load news and sentiment data"""
    return {
        "news": call_api("/api/news/feed?limit=20"),
        "sentiment": call_api("/api/news/sentiment")
    }

data = load_news_data()
news_articles = data.get("news", [])
sentiment_data = data.get("sentiment", {})

# Quick overview
st.markdown('<div class="section-header">📊 Market Overview</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("News Articles", len(news_articles))

with col2:
    overall_sentiment = sentiment_data.get("overall_sentiment", 0)
    sentiment_label = "🟢 Positive" if overall_sentiment > 0.1 else "🔴 Negative" if overall_sentiment < -0.1 else "🟡 Neutral"
    st.metric("Market Sentiment", sentiment_label)

with col3:
    confidence = sentiment_data.get("sentiment_confidence", 0)
    st.metric("Confidence", f"{confidence:.0%}")

with col4:
    # Count recent articles (last 6 hours)
    current_time = datetime.now()
    recent_count = 0
    for article in news_articles:
        try:
            pub_time = datetime.fromisoformat(article['published_at'].replace('Z', '+00:00')).replace(tzinfo=None)
            if (current_time - pub_time).total_seconds() < 21600:  # 6 hours
                recent_count += 1
        except:
            pass
    
    st.metric("Recent (6h)", recent_count)

# Simple filters
st.markdown("---")
st.markdown('<div class="section-header">🔍 Filters</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    # Extract unique symbols
    all_symbols = set()
    for article in news_articles:
        all_symbols.update(article.get('symbols', []))
    
    symbol_filter = st.selectbox(
        "Filter by Currency Pair",
        options=["All"] + sorted(list(all_symbols)),
        index=0
    )

with col2:
    sentiment_filter = st.selectbox(
        "Filter by Sentiment",
        options=["All", "Positive", "Negative", "Neutral"],
        index=0
    )

with col3:
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Apply filters
filtered_articles = news_articles.copy()

if symbol_filter != "All":
    filtered_articles = [a for a in filtered_articles if symbol_filter in a.get('symbols', [])]

if sentiment_filter != "All":
    if sentiment_filter == "Positive":
        filtered_articles = [a for a in filtered_articles if a.get('sentiment', 0) > 0.1]
    elif sentiment_filter == "Negative":
        filtered_articles = [a for a in filtered_articles if a.get('sentiment', 0) < -0.1]
    elif sentiment_filter == "Neutral":
        filtered_articles = [a for a in filtered_articles if -0.1 <= a.get('sentiment', 0) <= 0.1]

# News feed
st.markdown("---")
st.markdown('<div class="section-header">📰 Latest News</div>', unsafe_allow_html=True)

if filtered_articles:
    for article in filtered_articles[:10]:  # Show top 10 articles
        # Format published time
        try:
            pub_time = datetime.fromisoformat(article['published_at'].replace('Z', '+00:00'))
            time_str = pub_time.strftime("%m/%d %H:%M")
        except:
            time_str = "Unknown"
        
        # Determine sentiment display
        sentiment = article.get('sentiment', 0)
        if sentiment > 0.1:
            sentiment_class = "sentiment-positive"
            sentiment_text = f"😊 Positive ({sentiment:.2f})"
        elif sentiment < -0.1:
            sentiment_class = "sentiment-negative"
            sentiment_text = f"😞 Negative ({sentiment:.2f})"
        else:
            sentiment_class = "sentiment-neutral"
            sentiment_text = f"😐 Neutral ({sentiment:.2f})"
        
        # Get symbols for display
        symbols = article.get('symbols', [])
        symbols_text = ", ".join(symbols[:3]) if symbols else "General"
        
        # News card
        st.markdown(f"""
        <div class="news-card">
            <div class="news-title-text">{article.get('title', 'No Title')}</div>
            <div class="news-meta">
                📅 {time_str} | 📰 {article.get('source', 'Unknown')} | 💱 {symbols_text}
            </div>
            <div class="news-summary">{article.get('summary', 'No summary available')}</div>
            <div>
                <span class="{sentiment_class}">{sentiment_text}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Load more button
    if len(news_articles) > 10:
        st.markdown("---")
        if st.button(f"📄 Show {min(10, len(news_articles) - 10)} More Articles", use_container_width=True):
            # In a real implementation, this would load more articles
            st.info("More articles would be loaded here")

else:
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #6c757d;">
        <h4>📰 No Articles Found</h4>
        <p>No articles match your current filter criteria. Try adjusting the filters or refresh the data.</p>
    </div>
    """, unsafe_allow_html=True)

# Market sentiment summary
st.markdown("---")
st.markdown('<div class="section-header">📊 Market Sentiment Summary</div>', unsafe_allow_html=True)

if sentiment_data:
    trending_positive = sentiment_data.get("trending_positive", [])
    trending_negative = sentiment_data.get("trending_negative", [])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Trending Positive")
        if trending_positive:
            for symbol in trending_positive:
                st.markdown(f"🟢 **{symbol}** - Positive news sentiment")
        else:
            st.markdown("No strongly positive trends detected")
    
    with col2:
        st.markdown("### 📉 Trending Negative")
        if trending_negative:
            for symbol in trending_negative:
                st.markdown(f"🔴 **{symbol}** - Negative news sentiment")
        else:
            st.markdown("No strongly negative trends detected")

# Quick navigation
st.markdown("---")
st.markdown('<div class="section-header">⚡ Quick Actions</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📈 View Signals", use_container_width=True):
        st.switch_page("pages/1_overview.py")

with col2:
    if st.button("⚙️ Strategy Settings", use_container_width=True):
        st.switch_page("pages/2_strategies.py")

with col3:
    if st.button("🛡️ Risk Management", use_container_width=True):
        st.switch_page("pages/3_risk.py")

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #7f8c8d; font-size: 0.9rem;">
    News data last updated: {datetime.now().strftime('%H:%M:%S')} | Updates every 5 minutes
</div>
""", unsafe_allow_html=True)