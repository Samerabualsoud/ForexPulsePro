"""
KPI Definitions Page - Comprehensive Key Performance Indicators Guide
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

st.set_page_config(page_title="KPI Definitions", page_icon="📊", layout="wide")

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from utils.auth import require_authentication, render_user_info
    from utils.cache import get_cached_performance_stats, get_cached_market_data
    
    # Require authentication for this page
    user_info = require_authentication()
    render_user_info()
    imports_available = True
except ImportError as e:
    st.warning(f"⚠️ Import error: {e} - running in demo mode")
    user_info = {"username": "demo", "role": "admin"}
    imports_available = False

# Professional KPI Definitions Page Styling
st.markdown("""
<style>
    .kpi-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.8rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .kpi-section {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
    }
    
    .kpi-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 1.8rem;
        border-radius: 15px;
        border: 2px solid #e1e8ed;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        transition: transform 0.2s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.15);
    }
    
    .kpi-name {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.4rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .kpi-formula {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
        font-weight: bold;
        color: #2d3436;
        border-left: 4px solid #fdcb6e;
    }
    
    .kpi-example {
        background: linear-gradient(135deg, #a8e6cf 0%, #88d8c0 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #2d3436;
        border-left: 4px solid #00b894;
    }
    
    .kpi-range {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #2d3436;
        border-left: 4px solid #17a2b8;
    }
    
    .search-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .category-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.8rem;
        font-weight: bold;
        margin: 2rem 0 1rem 0;
        text-align: center;
    }
    
    .importance-high {
        border-left: 5px solid #e74c3c;
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
    }
    
    .importance-medium {
        border-left: 5px solid #f39c12;
        background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);
    }
    
    .importance-low {
        border-left: 5px solid #3498db;
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
    }
</style>
""", unsafe_allow_html=True)

# Page Title
st.markdown('<h1 class="kpi-title">📊 KPI Definitions & Trading Metrics</h1>', unsafe_allow_html=True)
st.markdown("### *Comprehensive Guide to Key Performance Indicators*")
st.markdown("---")

# Search functionality
st.markdown('<div class="search-box">', unsafe_allow_html=True)
st.markdown("### 🔍 **Find KPI Definitions**")
search_term = st.text_input("Search for a specific KPI or trading metric:", placeholder="e.g., Sharpe Ratio, Win Rate, Drawdown...")
category_filter = st.selectbox(
    "Filter by category:",
    ["All Categories", "Return Metrics", "Risk Metrics", "Performance Ratios", "Volume & Activity", "Advanced Metrics"]
)
st.markdown('</div>', unsafe_allow_html=True)

# Define comprehensive KPI definitions
kpi_definitions = {
    "Return Metrics": {
        "Total P&L": {
            "definition": "The total profit or loss generated from all trading activities over a specific period.",
            "formula": "Total P&L = Sum of all (Exit Price - Entry Price) × Position Size",
            "example": "If you make $500 on Trade A and lose $200 on Trade B, Total P&L = $300",
            "ideal_range": "Positive values indicate profitability. Target: >5% monthly returns",
            "importance": "high",
            "description": "The most fundamental measure of trading success. Shows absolute performance but doesn't account for risk taken."
        },
        "Average P&L": {
            "definition": "The average profit or loss per trade, calculated across all completed trades.",
            "formula": "Average P&L = Total P&L ÷ Number of Trades",
            "example": "Total P&L of $1,000 across 20 trades = $50 average P&L per trade",
            "ideal_range": "Positive values preferred. Good: >$25 per trade for retail",
            "importance": "high",
            "description": "Indicates typical trade performance. Essential for position sizing and expectancy calculations."
        },
        "Return on Investment (ROI)": {
            "definition": "The percentage return on the initial capital investment over a specific period.",
            "formula": "ROI = (Current Value - Initial Investment) ÷ Initial Investment × 100%",
            "example": "$10,000 account grows to $12,000 = 20% ROI",
            "ideal_range": "Excellent: >20% annually, Good: 10-20%, Average: 5-10%",
            "importance": "high",
            "description": "Shows efficiency of capital utilization. Must be risk-adjusted for meaningful comparison."
        }
    },
    
    "Performance Ratios": {
        "Win Rate": {
            "definition": "The percentage of trades that resulted in a profit out of total trades executed.",
            "formula": "Win Rate = (Number of Winning Trades ÷ Total Trades) × 100%",
            "example": "8 winning trades out of 10 total trades = 80% win rate",
            "ideal_range": "Excellent: >70%, Good: 50-70%, Acceptable: >40%",
            "importance": "high",
            "description": "Higher win rates are generally preferred but must be balanced with profit factor. A 90% win rate with small wins and large losses can be unprofitable."
        },
        "Profit Factor": {
            "definition": "The ratio of total profits to total losses, indicating overall profitability.",
            "formula": "Profit Factor = Total Gross Profit ÷ Total Gross Loss",
            "example": "$5,000 in profits and $2,000 in losses = 2.5 profit factor",
            "ideal_range": "Excellent: >2.0, Good: 1.5-2.0, Breakeven: 1.0, Unprofitable: <1.0",
            "importance": "high",
            "description": "Values above 1.0 indicate profitability. A profit factor of 2.0 means you make $2 for every $1 lost."
        },
        "Sharpe Ratio": {
            "definition": "Risk-adjusted return measure that compares excess return to volatility.",
            "formula": "Sharpe Ratio = (Portfolio Return - Risk-Free Rate) ÷ Portfolio Standard Deviation",
            "example": "12% return, 3% risk-free rate, 15% volatility = (12-3)/15 = 0.6 Sharpe",
            "ideal_range": "Excellent: >1.5, Good: 1.0-1.5, Acceptable: 0.5-1.0, Poor: <0.5",
            "importance": "high",
            "description": "Higher values indicate better risk-adjusted performance. Accounts for volatility in returns."
        },
        "Calmar Ratio": {
            "definition": "Risk-adjusted return measure that compares annual return to maximum drawdown.",
            "formula": "Calmar Ratio = Annual Return ÷ Maximum Drawdown",
            "example": "15% annual return with 10% max drawdown = 1.5 Calmar ratio",
            "ideal_range": "Excellent: >2.0, Good: 1.0-2.0, Acceptable: 0.5-1.0",
            "importance": "medium",
            "description": "Focuses specifically on downside risk. Preferred by institutional investors for tail risk assessment."
        }
    },
    
    "Risk Metrics": {
        "Maximum Drawdown": {
            "definition": "The largest peak-to-trough decline in account value during a specific period.",
            "formula": "Max Drawdown = (Peak Value - Trough Value) ÷ Peak Value × 100%",
            "example": "Account drops from $10,000 to $8,500 = 15% maximum drawdown",
            "ideal_range": "Excellent: <5%, Good: 5-10%, Acceptable: 10-20%, Risky: >20%",
            "importance": "high",
            "description": "Critical risk measure. Large drawdowns can be psychologically and financially devastating."
        },
        "Volatility": {
            "definition": "The degree of variation in trading returns, measured as standard deviation.",
            "formula": "Volatility = Standard Deviation of Returns × √(252) for annualized",
            "example": "Daily returns vary ±2% typically = ~32% annualized volatility",
            "ideal_range": "Conservative: <15%, Moderate: 15-25%, Aggressive: >25%",
            "importance": "high",
            "description": "Higher volatility means more unpredictable returns. Must match risk tolerance."
        },
        "Value at Risk (VaR)": {
            "definition": "The maximum expected loss over a specific time period at a given confidence level.",
            "formula": "VaR = Portfolio Value × (Expected Return - Z-score × Volatility)",
            "example": "95% confidence: Maximum daily loss of $500 on $10,000 portfolio",
            "ideal_range": "Should not exceed 2-5% of capital daily",
            "importance": "medium",
            "description": "Provides concrete dollar risk figures for risk management decisions."
        },
        "Beta": {
            "definition": "Measures how much a trading strategy moves relative to the overall market.",
            "formula": "Beta = Covariance(Strategy, Market) ÷ Variance(Market)",
            "example": "Beta of 1.2 means strategy moves 20% more than market",
            "ideal_range": "Market neutral: ~0, Conservative: 0-0.8, Aggressive: >1.2",
            "importance": "medium",
            "description": "Indicates correlation with broader market movements. Important for portfolio diversification."
        }
    },
    
    "Volume & Activity": {
        "Total Trades": {
            "definition": "The total number of completed trades executed during a specific period.",
            "formula": "Total Trades = Count of all Entry and Exit pairs",
            "example": "Executed 45 complete trades in one month",
            "ideal_range": "Depends on strategy: Scalping >100/day, Swing 5-20/month",
            "importance": "medium",
            "description": "More trades provide better statistical significance but increase transaction costs."
        },
        "Average Trade Duration": {
            "definition": "The average time between entering and exiting trades.",
            "formula": "Avg Duration = Sum of all Trade Durations ÷ Number of Trades",
            "example": "Total 120 hours across 24 trades = 5 hours average duration",
            "ideal_range": "Scalping: <1 hour, Intraday: 1-8 hours, Swing: 1-7 days",
            "importance": "low",
            "description": "Indicates trading style and strategy type. Affects capital efficiency and overnight risk."
        },
        "Trade Frequency": {
            "definition": "The number of trades executed per unit of time (daily, weekly, monthly).",
            "formula": "Trade Frequency = Number of Trades ÷ Time Period",
            "example": "60 trades in 30 days = 2 trades per day frequency",
            "ideal_range": "Should match strategy design and market opportunities",
            "importance": "low",
            "description": "Higher frequency can mean more opportunities but also higher costs and complexity."
        }
    },
    
    "Advanced Metrics": {
        "Alpha": {
            "definition": "The excess return of a strategy compared to the return of a benchmark index.",
            "formula": "Alpha = Strategy Return - (Risk-Free Rate + Beta × (Market Return - Risk-Free Rate))",
            "example": "Strategy returns 15%, market 10%, beta 1.2, risk-free 3% = 3.6% alpha",
            "ideal_range": "Positive alpha indicates outperformance. Target: >2% annually",
            "importance": "medium",
            "description": "Shows true skill in generating returns above market expectations."
        },
        "Information Ratio": {
            "definition": "Risk-adjusted measure of active return versus tracking error relative to benchmark.",
            "formula": "Information Ratio = (Portfolio Return - Benchmark Return) ÷ Tracking Error",
            "example": "5% outperformance with 8% tracking error = 0.625 information ratio",
            "ideal_range": "Excellent: >0.5, Good: 0.25-0.5, Poor: <0.25",
            "importance": "medium",
            "description": "Measures consistency of outperformance. Higher values indicate more reliable alpha generation."
        },
        "Sortino Ratio": {
            "definition": "Risk-adjusted return measure that only considers downside volatility.",
            "formula": "Sortino Ratio = (Return - Target) ÷ Downside Deviation",
            "example": "12% return, 8% target, 10% downside deviation = 0.4 Sortino ratio",
            "ideal_range": "Excellent: >1.5, Good: 1.0-1.5, Acceptable: 0.5-1.0",
            "importance": "medium",
            "description": "Better than Sharpe for asymmetric return distributions. Focuses only on bad volatility."
        },
        "Omega Ratio": {
            "definition": "Performance measure that captures all moments of the return distribution.",
            "formula": "Omega = Probability Weighted Gains ÷ Probability Weighted Losses",
            "example": "Omega of 1.3 means expected gains are 1.3x expected losses",
            "ideal_range": "Excellent: >1.5, Good: 1.2-1.5, Breakeven: 1.0",
            "importance": "low",
            "description": "Comprehensive measure that doesn't assume normal distribution of returns."
        }
    }
}

# Filter KPIs based on search and category
def filter_kpis(kpis, search_term, category_filter):
    filtered_kpis = {}
    
    for category, kpi_list in kpis.items():
        if category_filter != "All Categories" and category != category_filter:
            continue
            
        filtered_category = {}
        for kpi_name, kpi_data in kpi_list.items():
            # Search in KPI name, definition, and description
            search_text = f"{kpi_name} {kpi_data['definition']} {kpi_data['description']}".lower()
            if not search_term or search_term.lower() in search_text:
                filtered_category[kpi_name] = kpi_data
        
        if filtered_category:
            filtered_kpis[category] = filtered_category
    
    return filtered_kpis

# Apply filters
filtered_kpis = filter_kpis(kpi_definitions, search_term, category_filter)

# Display summary statistics
st.markdown('<div class="kpi-section">', unsafe_allow_html=True)
st.markdown("### 📈 **KPI Overview**")

col1, col2, col3, col4 = st.columns(4)

total_kpis = sum(len(kpi_list) for kpi_list in kpi_definitions.values())
filtered_count = sum(len(kpi_list) for kpi_list in filtered_kpis.values())
high_importance = sum(1 for category in kpi_definitions.values() 
                     for kpi in category.values() if kpi['importance'] == 'high')

with col1:
    st.metric("Total KPIs", f"{total_kpis}", delta="Comprehensive coverage")

with col2:
    st.metric("Showing", f"{filtered_count}", delta=f"Filtered from {total_kpis}")

with col3:
    st.metric("High Priority", f"{high_importance}", delta="Essential metrics")

with col4:
    st.metric("Categories", f"{len(kpi_definitions)}", delta="Complete coverage")

st.markdown('</div>', unsafe_allow_html=True)

# Display filtered KPIs
if not filtered_kpis:
    st.warning("🔍 No KPIs match your search criteria. Try a different search term or category.")
else:
    for category, kpi_list in filtered_kpis.items():
        st.markdown(f'<h2 class="category-header">{category}</h2>', unsafe_allow_html=True)
        
        for kpi_name, kpi_data in kpi_list.items():
            # Determine card styling based on importance
            importance_class = f"importance-{kpi_data['importance']}"
            
            st.markdown('<div class="kpi-section">', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-card {importance_class}">', unsafe_allow_html=True)
            
            # KPI Name and Definition
            st.markdown(f'<h3 class="kpi-name">{kpi_name}</h3>', unsafe_allow_html=True)
            st.markdown(f"**Definition:** {kpi_data['definition']}")
            st.markdown(f"**Purpose:** {kpi_data['description']}")
            
            # Formula
            st.markdown(f'<div class="kpi-formula"><strong>📐 Formula:</strong><br>{kpi_data["formula"]}</div>', unsafe_allow_html=True)
            
            # Example
            st.markdown(f'<div class="kpi-example"><strong>💡 Example:</strong><br>{kpi_data["example"]}</div>', unsafe_allow_html=True)
            
            # Ideal Range
            st.markdown(f'<div class="kpi-range"><strong>🎯 Ideal Range:</strong><br>{kpi_data["ideal_range"]}</div>', unsafe_allow_html=True)
            
            # Importance indicator
            importance_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}
            st.markdown(f"**Priority Level:** {importance_emoji[kpi_data['importance']]} {kpi_data['importance'].title()}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# Quick Reference Section
st.markdown("---")
st.markdown('<div class="kpi-section">', unsafe_allow_html=True)
st.markdown("### 🚀 **Quick Reference Guide**")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🎯 **Essential KPIs for Beginners**")
    beginner_kpis = [
        "📊 **Win Rate** - Start here to understand success rate",
        "💰 **Total P&L** - Track overall profitability",
        "📉 **Maximum Drawdown** - Monitor worst-case scenarios",
        "⚖️ **Profit Factor** - Ensure profits exceed losses",
        "📈 **Average P&L** - Know your typical trade outcome"
    ]
    for kpi in beginner_kpis:
        st.write(kpi)

with col2:
    st.markdown("#### 🏆 **Advanced KPIs for Professionals**")
    advanced_kpis = [
        "📊 **Sharpe Ratio** - Risk-adjusted performance",
        "🎯 **Alpha** - Market outperformance measurement",
        "📉 **Sortino Ratio** - Downside-focused risk adjustment",
        "📈 **Information Ratio** - Consistent alpha generation",
        "⚡ **Calmar Ratio** - Drawdown-adjusted returns"
    ]
    for kpi in advanced_kpis:
        st.write(kpi)

# Best Practices
st.markdown("---")
st.markdown("#### 💡 **KPI Analysis Best Practices**")

practices = [
    "🔄 **Regular Monitoring**: Review KPIs weekly for short-term strategies, monthly for long-term",
    "📊 **Benchmark Comparison**: Always compare performance against relevant market indices",
    "⚖️ **Risk-Return Balance**: Never evaluate returns without considering risk metrics",
    "📈 **Trend Analysis**: Look at KPI trends over time, not just point-in-time values",
    "🎯 **Goal Setting**: Set realistic KPI targets based on your experience and capital",
    "📋 **Documentation**: Keep detailed records of KPI changes and their causes",
    "🔍 **Multiple Metrics**: Use combinations of KPIs for comprehensive performance assessment"
]

for practice in practices:
    st.write(practice)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("*This comprehensive guide covers the most important KPIs used in forex trading and algorithmic strategies. Use these metrics to objectively evaluate and improve your trading performance.*")