"""
News Feed Component
"""
import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import plotly.express as px
import plotly.graph_objects as go

# Import sentiment indicator component
from .sentiment_indicator import render_sentiment_indicator, render_sentiment_summary, format_sentiment_text

def render_news_feed(
    news_articles: List[Dict[str, Any]], 
    title: str = "Latest News",
    show_sentiment: bool = True,
    show_filters: bool = True,
    max_articles: Optional[int] = None,
    compact_view: bool = False
) -> None:
    """
    Render a comprehensive news feed with sentiment indicators
    
    Args:
        news_articles: List of news article dictionaries
        title: Feed title
        show_sentiment: Whether to show sentiment indicators
        show_filters: Whether to show filtering options
        max_articles: Maximum number of articles to display
        compact_view: Whether to use compact layout
    """
    
    if not news_articles:
        st.info(f"No {title.lower()} available")
        return
    
    # Apply max articles limit
    display_articles = news_articles[:max_articles] if max_articles else news_articles
    
    # Enhanced CSS styling for news feed
    st.markdown("""
    <style>
        .news-article {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
            border-left: 4px solid #667eea;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .news-article:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        }
        
        .news-title {
            font-size: 1.2rem;
            font-weight: bold;
            color: #2d3436;
            margin-bottom: 0.5rem;
            line-height: 1.4;
        }
        
        .news-summary {
            color: #636e72;
            margin-bottom: 1rem;
            line-height: 1.5;
            font-size: 0.95rem;
        }
        
        .news-meta {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.85rem;
            color: #74b9ff;
            margin-bottom: 1rem;
        }
        
        .news-source {
            font-weight: 600;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .news-time {
            color: #636e72;
            font-style: italic;
        }
        
        .sentiment-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid #ddd;
        }
        
        .symbols-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 0.3rem;
            margin-top: 0.5rem;
        }
        
        .symbol-tag {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 0.2rem 0.6rem;
            border-radius: 15px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        
        .compact-article {
            background: #f8f9fa;
            padding: 0.8rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            border-left: 3px solid #667eea;
        }
        
        .compact-title {
            font-size: 1rem;
            font-weight: 600;
            color: #2d3436;
            margin-bottom: 0.3rem;
        }
        
        .compact-meta {
            font-size: 0.8rem;
            color: #636e72;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.subheader(f"📰 {title}")
    
    # Filtering options
    if show_filters:
        render_news_filters(news_articles)
    
    # Display articles
    for i, article in enumerate(display_articles):
        if compact_view:
            render_compact_article(article, show_sentiment)
        else:
            render_full_article(article, show_sentiment)

def render_full_article(article: Dict[str, Any], show_sentiment: bool = True) -> None:
    """Render a full news article with all details"""
    
    # Format timestamp
    published_time = "Unknown"
    if article.get('published_at'):
        try:
            dt = datetime.fromisoformat(article['published_at'].replace('Z', '+00:00'))
            published_time = dt.strftime("%Y-%m-%d %H:%M")
        except (ValueError, TypeError):
            published_time = str(article['published_at'])[:19]
    
    # Extract data
    title = article.get('title', 'No Title')
    summary = article.get('summary', article.get('description', 'No summary available'))
    source = article.get('source', 'Unknown Source')
    url = article.get('url', '#')
    symbols = article.get('symbols', [])
    sentiment_score = article.get('sentiment_score', 0)
    sentiment_confidence = article.get('sentiment_confidence', 0)
    
    # Create article HTML
    article_html = f"""
    <div class="news-article">
        <div class="news-title">{title}</div>
        
        <div class="news-meta">
            <span class="news-source">📰 {source}</span>
            <span class="news-time">🕒 {published_time}</span>
        </div>
        
        <div class="news-summary">{summary}</div>
        
        {render_symbols_tags(symbols)}
        
        <div class="sentiment-row">
            <div>
                <a href="{url}" target="_blank" style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 0.4rem 0.8rem;
                    border-radius: 8px;
                    text-decoration: none;
                    font-weight: 600;
                    font-size: 0.9rem;
                ">📖 Read More</a>
            </div>
            {f'<div>{render_inline_sentiment(sentiment_score, sentiment_confidence)}</div>' if show_sentiment else ''}
        </div>
    </div>
    """
    
    st.markdown(article_html, unsafe_allow_html=True)
    
    # Display sentiment indicator if enabled
    if show_sentiment and sentiment_score != 0:
        col1, col2, col3 = st.columns([2, 1, 1])
        with col2:
            render_sentiment_indicator(sentiment_score, sentiment_confidence, size="small")

def render_compact_article(article: Dict[str, Any], show_sentiment: bool = True) -> None:
    """Render a compact news article for sidebars or widgets"""
    
    # Format timestamp
    published_time = "Unknown"
    if article.get('published_at'):
        try:
            dt = datetime.fromisoformat(article['published_at'].replace('Z', '+00:00'))
            published_time = dt.strftime("%m/%d %H:%M")
        except (ValueError, TypeError):
            published_time = str(article['published_at'])[:16]
    
    title = article.get('title', 'No Title')
    source = article.get('source', 'Unknown')
    url = article.get('url', '#')
    sentiment_score = article.get('sentiment_score', 0)
    sentiment_confidence = article.get('sentiment_confidence', 0)
    
    # Create compact article HTML
    compact_html = f"""
    <div class="compact-article">
        <div class="compact-title">{title[:80]}{'...' if len(title) > 80 else ''}</div>
        <div class="compact-meta">
            <span>📰 {source} • 🕒 {published_time}</span>
            <span>
                <a href="{url}" target="_blank" style="color: #667eea; text-decoration: none;">📖</a>
                {render_inline_sentiment(sentiment_score, sentiment_confidence, compact=True) if show_sentiment else ''}
            </span>
        </div>
    </div>
    """
    
    st.markdown(compact_html, unsafe_allow_html=True)

def render_symbols_tags(symbols: List[str]) -> str:
    """Render symbol tags HTML"""
    if not symbols:
        return ""
    
    tags_html = '<div class="symbols-tags">'
    for symbol in symbols[:5]:  # Limit to 5 symbols
        tags_html += f'<span class="symbol-tag">{symbol}</span>'
    if len(symbols) > 5:
        tags_html += f'<span class="symbol-tag">+{len(symbols) - 5} more</span>'
    tags_html += '</div>'
    
    return tags_html

def render_inline_sentiment(sentiment_score: float, confidence: float = 0, compact: bool = False) -> str:
    """Render inline sentiment indicator HTML"""
    
    if sentiment_score > 0.1:
        emoji = "😊"
        color = "#28a745"
    elif sentiment_score < -0.1:
        emoji = "😞"
        color = "#dc3545"
    else:
        emoji = "😐"
        color = "#6c757d"
    
    if compact:
        return f'<span style="font-size: 1.2rem;">{emoji}</span>'
    else:
        confidence_text = f" ({confidence:.1%})" if confidence > 0 else ""
        return f'<span style="color: {color}; font-weight: bold;">{emoji} {sentiment_score:.2f}{confidence_text}</span>'

def render_news_filters(news_articles: List[Dict[str, Any]]) -> None:
    """Render filtering options for news feed"""
    
    # Extract unique symbols and sources
    all_symbols = set()
    all_sources = set()
    
    for article in news_articles:
        symbols = article.get('symbols', [])
        all_symbols.update(symbols)
        if article.get('source'):
            all_sources.add(article['source'])
    
    # Filter controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        selected_symbols = st.multiselect(
            "Filter by Symbols:",
            options=sorted(list(all_symbols)),
            key="news_symbol_filter"
        )
    
    with col2:
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
    
    # Apply filters (this would be used in the main page logic)
    st.session_state.news_filters = {
        'symbols': selected_symbols,
        'sources': selected_sources,
        'sentiment': sentiment_filter,
        'time_range': time_filter
    }

def render_news_summary_widget(news_articles: List[Dict[str, Any]], max_items: int = 5) -> None:
    """Render a compact news summary widget for dashboards"""
    
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
    ">
        <h3 style="margin: 0 0 1rem 0; color: white;">📰 Latest Market News</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if not news_articles:
        st.info("No recent news available")
        return
    
    # Show most recent articles
    recent_articles = news_articles[:max_items]
    
    for article in recent_articles:
        render_compact_article(article, show_sentiment=True)
    
    # Show more button
    if len(news_articles) > max_items:
        if st.button(f"📖 View All {len(news_articles)} Articles", use_container_width=True):
            st.switch_page("pages/7_news.py")

def render_market_sentiment_widget(sentiment_data: List[Dict[str, Any]]) -> None:
    """Render market sentiment summary widget"""
    
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: #2d3436;
    ">
        <h3 style="margin: 0 0 1rem 0; color: #2d3436;">📊 Market Sentiment</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if sentiment_data:
        render_sentiment_summary(sentiment_data, title="")
    else:
        st.info("No sentiment data available")

def render_news_analytics(news_articles: List[Dict[str, Any]]) -> None:
    """Render news analytics charts and insights"""
    
    if not news_articles:
        st.info("No data available for analytics")
        return
    
    st.subheader("📊 News Analytics")
    
    # Convert to DataFrame for analysis
    df_data = []
    for article in news_articles:
        published_at = article.get('published_at')
        if published_at:
            try:
                dt = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                df_data.append({
                    'timestamp': dt,
                    'sentiment': article.get('sentiment_score', 0),
                    'source': article.get('source', 'Unknown'),
                    'symbols': len(article.get('symbols', [])),
                    'confidence': article.get('sentiment_confidence', 0)
                })
            except (ValueError, TypeError):
                continue
    
    if not df_data:
        st.info("No valid timestamp data for analytics")
        return
    
    df = pd.DataFrame(df_data)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment over time
        fig_timeline = px.line(
            df, 
            x='timestamp', 
            y='sentiment',
            title='Sentiment Timeline',
            color_discrete_sequence=['#667eea']
        )
        fig_timeline.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_timeline.update_layout(height=300)
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    with col2:
        # Source distribution
        source_counts = df['source'].value_counts().head(10)
        fig_sources = px.bar(
            x=source_counts.values,
            y=source_counts.index,
            orientation='h',
            title='News Sources',
            color_discrete_sequence=['#764ba2']
        )
        fig_sources.update_layout(height=300)
        st.plotly_chart(fig_sources, use_container_width=True)
    
    # Additional metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_sentiment = df['sentiment'].mean()
        st.metric("Avg Sentiment", f"{avg_sentiment:.3f}")
    
    with col2:
        total_articles = len(df)
        st.metric("Total Articles", total_articles)
    
    with col3:
        avg_confidence = df['confidence'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.2%}")
    
    with col4:
        unique_sources = df['source'].nunique()
        st.metric("Unique Sources", unique_sources)

def apply_news_filters(
    news_articles: List[Dict[str, Any]], 
    filters: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Apply filters to news articles"""
    
    filtered_articles = news_articles.copy()
    
    # Symbol filter
    if filters.get('symbols'):
        filtered_articles = [
            article for article in filtered_articles
            if any(symbol in article.get('symbols', []) for symbol in filters['symbols'])
        ]
    
    # Source filter
    if filters.get('sources'):
        filtered_articles = [
            article for article in filtered_articles
            if article.get('source') in filters['sources']
        ]
    
    # Sentiment filter
    sentiment_filter = filters.get('sentiment', 'All')
    if sentiment_filter != 'All':
        if sentiment_filter == 'Positive':
            filtered_articles = [a for a in filtered_articles if a.get('sentiment_score', 0) > 0.1]
        elif sentiment_filter == 'Negative':
            filtered_articles = [a for a in filtered_articles if a.get('sentiment_score', 0) < -0.1]
        elif sentiment_filter == 'Neutral':
            filtered_articles = [a for a in filtered_articles if -0.1 <= a.get('sentiment_score', 0) <= 0.1]
    
    # Time filter
    time_filter = filters.get('time_range', 'All')
    if time_filter != 'All':
        cutoff_time = datetime.now()
        if time_filter == 'Last Hour':
            cutoff_time -= timedelta(hours=1)
        elif time_filter == 'Last 6 Hours':
            cutoff_time -= timedelta(hours=6)
        elif time_filter == 'Last 24 Hours':
            cutoff_time -= timedelta(days=1)
        elif time_filter == 'Last Week':
            cutoff_time -= timedelta(weeks=1)
        
        filtered_articles = [
            article for article in filtered_articles
            if article.get('published_at') and 
            datetime.fromisoformat(article['published_at'].replace('Z', '+00:00')) >= cutoff_time
        ]
    
    return filtered_articles