"""
Sentiment Indicator Component
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List, Optional
from datetime import datetime

def render_sentiment_indicator(
    sentiment_score: float,
    confidence: float = 0.0,
    size: str = "medium",
    show_score: bool = True,
    show_confidence: bool = True
) -> None:
    """
    Render a visual sentiment indicator
    
    Args:
        sentiment_score: Sentiment score from -1 (negative) to +1 (positive)
        confidence: Confidence level from 0 to 1
        size: Size of indicator ("small", "medium", "large")
        show_score: Whether to show numeric score
        show_confidence: Whether to show confidence level
    """
    
    # Normalize sentiment score
    sentiment_score = max(-1, min(1, sentiment_score))
    confidence = max(0, min(1, confidence))
    
    # Determine sentiment category
    if sentiment_score > 0.1:
        sentiment_category = "positive"
        emoji = "😊"
        color = "#28a745"
        label = "Positive"
    elif sentiment_score < -0.1:
        sentiment_category = "negative"
        emoji = "😞"
        color = "#dc3545"
        label = "Negative"
    else:
        sentiment_category = "neutral"
        emoji = "😐"
        color = "#6c757d"
        label = "Neutral"
    
    # Size configurations
    size_configs = {
        "small": {"font_size": "1.2rem", "emoji_size": "1.5rem", "width": "120px"},
        "medium": {"font_size": "1.4rem", "emoji_size": "2rem", "width": "160px"},
        "large": {"font_size": "1.8rem", "emoji_size": "3rem", "width": "200px"}
    }
    
    config = size_configs.get(size, size_configs["medium"])
    
    # Create the indicator
    indicator_html = f"""
    <div style="
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0.5rem;
        border-radius: 10px;
        background: linear-gradient(135deg, {color}20, {color}10);
        border: 2px solid {color}40;
        width: {config['width']};
        margin: 0.2rem;
    ">
        <div style="
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
        ">
            <div style="font-size: {config['emoji_size']}; margin-bottom: 0.2rem;">
                {emoji}
            </div>
            <div style="
                font-size: {config['font_size']};
                font-weight: bold;
                color: {color};
                margin-bottom: 0.1rem;
            ">
                {label}
            </div>
            {f'<div style="font-size: 0.9rem; color: {color}80;">Score: {sentiment_score:.2f}</div>' if show_score else ''}
            {f'<div style="font-size: 0.8rem; color: {color}60;">Conf: {confidence:.1%}</div>' if show_confidence and confidence > 0 else ''}
        </div>
    </div>
    """
    
    st.markdown(indicator_html, unsafe_allow_html=True)

def render_sentiment_gauge(
    sentiment_score: float,
    title: str = "Market Sentiment",
    height: int = 300
) -> None:
    """
    Render a gauge chart for sentiment visualization
    
    Args:
        sentiment_score: Sentiment score from -1 to +1
        title: Title for the gauge
        height: Height of the chart
    """
    
    # Normalize sentiment score
    sentiment_score = max(-1, min(1, sentiment_score))
    
    # Convert to 0-100 scale for gauge
    gauge_value = (sentiment_score + 1) * 50
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = gauge_value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 24}},
        delta = {'reference': 50, 'relative': True, 'valueformat': '.1%'},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#ffcccc'},
                {'range': [30, 70], 'color': '#ffffcc'},
                {'range': [70, 100], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': gauge_value
            }
        }
    ))
    
    fig.update_layout(
        height=height,
        font={'color': "darkblue", 'family': "Arial"},
        margin={'l': 40, 'r': 40, 't': 40, 'b': 40}
    )
    
    st.plotly_chart(fig, use_container_width=True)

def normalize_sentiment_data(sentiment_data: Any) -> List[Dict[str, Any]]:
    """
    Normalize sentiment data to ensure consistent format
    
    Args:
        sentiment_data: Raw sentiment data (could be list, dict, string, or None)
        
    Returns:
        List of properly formatted sentiment dictionaries
    """
    if not sentiment_data:
        return []
    
    # If it's already a list, check each item
    if isinstance(sentiment_data, list):
        normalized_items = []
        for i, item in enumerate(sentiment_data):
            if isinstance(item, dict):
                # Already a dictionary, ensure it has required keys
                normalized_item = {
                    'symbol': item.get('symbol', f'Item {i+1}'),
                    'sentiment': float(item.get('sentiment', 0)) if item.get('sentiment') is not None else 0.0,
                    'confidence': float(item.get('confidence', 0)) if item.get('confidence') is not None else 0.0,
                    'article_count': int(item.get('article_count', 0)) if item.get('article_count') is not None else 0
                }
                normalized_items.append(normalized_item)
            elif isinstance(item, str):
                # String item, convert to basic sentiment entry
                normalized_items.append({
                    'symbol': item,
                    'sentiment': 0.0,
                    'confidence': 0.0,
                    'article_count': 0
                })
            elif isinstance(item, (int, float)):
                # Numeric item, treat as sentiment value
                normalized_items.append({
                    'symbol': f'Item {i+1}',
                    'sentiment': float(item),
                    'confidence': 0.0,
                    'article_count': 0
                })
            else:
                # Unknown type, create default entry
                normalized_items.append({
                    'symbol': f'Item {i+1}',
                    'sentiment': 0.0,
                    'confidence': 0.0,
                    'article_count': 0
                })
        return normalized_items
    
    # If it's a single dictionary, wrap in list
    elif isinstance(sentiment_data, dict):
        return normalize_sentiment_data([sentiment_data])
    
    # If it's a string, create a single entry
    elif isinstance(sentiment_data, str):
        return [{
            'symbol': sentiment_data,
            'sentiment': 0.0,
            'confidence': 0.0,
            'article_count': 0
        }]
    
    # For any other type, return empty list
    else:
        return []

def render_sentiment_summary(
    sentiment_data: Any,
    title: str = "Market Sentiment Overview"
) -> None:
    """
    Render a summary of sentiment data across multiple sources/symbols
    
    Args:
        sentiment_data: Sentiment data (any format - will be normalized)
        title: Title for the summary section
    """
    
    # Normalize the data to ensure consistent format
    normalized_data = normalize_sentiment_data(sentiment_data)
    
    if not normalized_data:
        st.info("No sentiment data available")
        return
    
    st.subheader(f"📊 {title}")
    
    # Calculate overall metrics with safe access
    total_items = len(normalized_data)
    try:
        avg_sentiment = sum(item.get('sentiment', 0) for item in normalized_data) / total_items
        avg_confidence = sum(item.get('confidence', 0) for item in normalized_data) / total_items
    except (TypeError, ZeroDivisionError):
        avg_sentiment = 0.0
        avg_confidence = 0.0
    
    # Calculate sentiment distribution with safe access
    try:
        positive_count = len([item for item in normalized_data if item.get('sentiment', 0) > 0.1])
        negative_count = len([item for item in normalized_data if item.get('sentiment', 0) < -0.1])
        neutral_count = total_items - positive_count - negative_count
    except (TypeError, AttributeError):
        positive_count = negative_count = neutral_count = 0
    
    # Display overall metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Overall Sentiment", f"{avg_sentiment:.2f}")
        render_sentiment_indicator(avg_sentiment, avg_confidence, size="small", show_confidence=False)
    
    with col2:
        st.metric("Positive", positive_count)
        positive_pct = (positive_count / total_items * 100) if total_items > 0 else 0
        st.markdown(f"<div style='color: #28a745; font-weight: bold;'>{positive_pct:.1f}%</div>", unsafe_allow_html=True)
    
    with col3:
        st.metric("Neutral", neutral_count)
        neutral_pct = (neutral_count / total_items * 100) if total_items > 0 else 0
        st.markdown(f"<div style='color: #6c757d; font-weight: bold;'>{neutral_pct:.1f}%</div>", unsafe_allow_html=True)
    
    with col4:
        st.metric("Negative", negative_count)
        negative_pct = (negative_count / total_items * 100) if total_items > 0 else 0
        st.markdown(f"<div style='color: #dc3545; font-weight: bold;'>{negative_pct:.1f}%</div>", unsafe_allow_html=True)
    
    # Individual sentiment indicators
    st.subheader("📈 By Symbol/Source")
    
    # Create columns for sentiment indicators - use normalized_data
    cols = st.columns(min(4, len(normalized_data)))
    
    for i, item in enumerate(normalized_data):
        col_idx = i % len(cols)
        with cols[col_idx]:
            # Safe access to normalized data
            symbol = item.get('symbol', f'Item {i+1}')
            sentiment = item.get('sentiment', 0)
            confidence = item.get('confidence', 0)
            
            st.markdown(f"**{symbol}**")
            render_sentiment_indicator(sentiment, confidence, size="small")

def render_sentiment_timeline(
    timeline_data: List[Dict[str, Any]],
    title: str = "Sentiment Timeline",
    height: int = 400
) -> None:
    """
    Render a timeline chart of sentiment changes
    
    Args:
        timeline_data: List of sentiment data with 'timestamp', 'sentiment', 'symbol'
        title: Title for the chart
        height: Height of the chart
    """
    
    if not timeline_data:
        st.info("No timeline data available")
        return
    
    st.subheader(f"📈 {title}")
    
    # Convert to DataFrame
    df = pd.DataFrame(timeline_data)
    
    if 'timestamp' not in df.columns or 'sentiment' not in df.columns:
        st.error("Invalid timeline data format")
        return
    
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create timeline chart
    if 'symbol' in df.columns:
        # Multi-line chart for different symbols
        fig = px.line(
            df,
            x='timestamp',
            y='sentiment',
            color='symbol',
            title=title,
            height=height,
            hover_data=['confidence'] if 'confidence' in df.columns else None
        )
    else:
        # Single line chart
        fig = px.line(
            df,
            x='timestamp',
            y='sentiment',
            title=title,
            height=height
        )
    
    # Add horizontal lines for neutral zones
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_hline(y=0.1, line_dash="dot", line_color="green", opacity=0.3)
    fig.add_hline(y=-0.1, line_dash="dot", line_color="red", opacity=0.3)
    
    # Update layout
    fig.update_layout(
        yaxis_title="Sentiment Score",
        xaxis_title="Time",
        yaxis=dict(range=[-1.1, 1.1]),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def format_sentiment_text(sentiment_score: float, confidence: float = 0.0) -> str:
    """
    Format sentiment data as readable text
    
    Args:
        sentiment_score: Sentiment score from -1 to +1
        confidence: Confidence level from 0 to 1
    
    Returns:
        Formatted text description
    """
    
    if sentiment_score > 0.3:
        intensity = "Very Positive"
    elif sentiment_score > 0.1:
        intensity = "Positive"
    elif sentiment_score > -0.1:
        intensity = "Neutral"
    elif sentiment_score > -0.3:
        intensity = "Negative"
    else:
        intensity = "Very Negative"
    
    confidence_text = ""
    if confidence > 0:
        if confidence > 0.8:
            confidence_text = " (High Confidence)"
        elif confidence > 0.6:
            confidence_text = " (Medium Confidence)"
        else:
            confidence_text = " (Low Confidence)"
    
    return f"{intensity} ({sentiment_score:.2f}){confidence_text}"