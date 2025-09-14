"""
Trading Reference - KPI Definitions & Terminology Guide
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

st.set_page_config(page_title="Trading Reference", page_icon="📖", layout="wide")

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from utils.cache import get_cached_performance_stats, get_cached_market_data
    
    # No authentication required
    user_info = {"username": "user", "role": "admin"}
    imports_available = True
except ImportError as e:
    st.warning(f"⚠️ Import error: {e} - running in demo mode")
    user_info = {"username": "user", "role": "admin"}
    imports_available = False

# Professional Reference Page Styling
st.markdown("""
<style>
    .reference-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.8rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .reference-section {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
    }
    
    .ref-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 1.8rem;
        border-radius: 15px;
        border: 2px solid #e1e8ed;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        transition: transform 0.2s ease;
    }
    
    .ref-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.15);
    }
    
    .ref-name {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.4rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .ref-formula {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
        font-weight: bold;
        color: #2d3436;
        border-left: 4px solid #fdcb6e;
    }
    
    .ref-example {
        background: linear-gradient(135deg, #a8e6cf 0%, #88d8c0 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #2d3436;
        border-left: 4px solid #00b894;
    }
    
    .ref-range {
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
    
    .tab-container {
        background: white;
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    }
    
    .importance-high, .difficulty-advanced {
        border-left: 5px solid #e74c3c;
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
    }
    
    .importance-medium, .difficulty-intermediate {
        border-left: 5px solid #f39c12;
        background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);
    }
    
    .importance-low, .difficulty-beginner {
        border-left: 5px solid #00b894;
        background: linear-gradient(135deg, #a8e6cf 0%, #88d8c0 100%);
    }
</style>
""", unsafe_allow_html=True)

# Page Title
st.markdown('<h1 class="reference-title">📖 Trading Reference Guide</h1>', unsafe_allow_html=True)
st.markdown("### *Complete KPI Definitions & Trading Terminology*")
st.markdown("---")

# Tab selection
tab1, tab2 = st.tabs(["📊 KPI Definitions", "📚 Trading Terminology"])

with tab1:
    st.markdown("### Key Performance Indicators for Trading Success")
    
    # Search functionality for KPIs
    st.markdown('<div class="search-box">', unsafe_allow_html=True)
    st.markdown("### 🔍 **Find KPI Definitions**")
    kpi_search = st.text_input("Search KPIs:", placeholder="e.g., Sharpe Ratio, Win Rate, Drawdown...", key="kpi_search")
    kpi_category = st.selectbox(
        "Filter by category:",
        ["All Categories", "Return Metrics", "Risk Metrics", "Performance Ratios", "Volume & Activity", "Advanced Metrics"],
        key="kpi_category"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # KPI definitions
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
            "Sortino Ratio": {
                "definition": "Risk-adjusted return measure that only considers downside volatility.",
                "formula": "Sortino Ratio = (Return - Target) ÷ Downside Deviation",
                "example": "12% return, 8% target, 10% downside deviation = 0.4 Sortino ratio",
                "ideal_range": "Excellent: >1.5, Good: 1.0-1.5, Acceptable: 0.5-1.0",
                "importance": "medium",
                "description": "Better than Sharpe for asymmetric return distributions. Focuses only on bad volatility."
            }
        }
    }

    # Filter KPIs function
    def filter_kpis(kpis, search_term, category_filter):
        filtered_kpis = {}
        
        for category, kpi_list in kpis.items():
            if category_filter != "All Categories" and category != category_filter:
                continue
                
            filtered_category = {}
            for kpi_name, kpi_data in kpi_list.items():
                search_text = f"{kpi_name} {kpi_data['definition']} {kpi_data['description']}".lower()
                if not search_term or search_term.lower() in search_text:
                    filtered_category[kpi_name] = kpi_data
            
            if filtered_category:
                filtered_kpis[category] = filtered_category
        
        return filtered_kpis

    # Apply KPI filters
    filtered_kpis = filter_kpis(kpi_definitions, kpi_search, kpi_category)

    # Display KPIs
    if not filtered_kpis:
        st.warning("🔍 No KPIs match your search criteria. Try a different search term or category.")
    else:
        for category, kpi_list in filtered_kpis.items():
            st.markdown(f'<h2 class="category-header">{category}</h2>', unsafe_allow_html=True)
            
            for kpi_name, kpi_data in kpi_list.items():
                importance_class = f"importance-{kpi_data['importance']}"
                
                st.markdown('<div class="reference-section">', unsafe_allow_html=True)
                st.markdown(f'<div class="ref-card {importance_class}">', unsafe_allow_html=True)
                
                st.markdown(f'<h3 class="ref-name">{kpi_name}</h3>', unsafe_allow_html=True)
                st.markdown(f"**Definition:** {kpi_data['definition']}")
                st.markdown(f"**Purpose:** {kpi_data['description']}")
                
                st.markdown(f'<div class="ref-formula"><strong>📐 Formula:</strong><br>{kpi_data["formula"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="ref-example"><strong>💡 Example:</strong><br>{kpi_data["example"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="ref-range"><strong>🎯 Ideal Range:</strong><br>{kpi_data["ideal_range"]}</div>', unsafe_allow_html=True)
                
                importance_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                st.markdown(f"**Priority Level:** {importance_emoji[kpi_data['importance']]} {kpi_data['importance'].title()}")
                
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown("### Trading Terms & Concepts Glossary")
    
    # Search functionality for terminology
    st.markdown('<div class="search-box">', unsafe_allow_html=True)
    st.markdown("### 🔍 **Find Trading Terms**")
    
    col1, col2 = st.columns(2)
    with col1:
        term_search = st.text_input("Search terms:", placeholder="e.g., Pip, Leverage, Moving Average...", key="term_search")
        term_category = st.selectbox(
            "Filter by category:",
            ["All Categories", "Basic Forex", "Technical Analysis", "Order Types", "Risk Management", "Market Structure", "Advanced Concepts"],
            key="term_category"
        )

    with col2:
        term_difficulty = st.selectbox(
            "Filter by difficulty:",
            ["All Levels", "Beginner", "Intermediate", "Advanced"],
            key="term_difficulty"
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # Terminology database
    terminology = {
        "Basic Forex": {
            "Pip": {
                "definition": "The smallest price move in a currency pair, typically the fourth decimal place (0.0001).",
                "example": "If EUR/USD moves from 1.1050 to 1.1051, that's a 1 pip movement.",
                "usage": "Used to measure price movements and calculate profits/losses in forex trading.",
                "difficulty": "beginner",
                "also_known_as": "Point in Percentage, Price Interest Point"
            },
            "Spread": {
                "definition": "The difference between the bid (sell) price and ask (buy) price of a currency pair.",
                "example": "If EUR/USD bid is 1.1050 and ask is 1.1052, the spread is 2 pips.",
                "usage": "Represents the broker's profit and trading cost for the trader.",
                "difficulty": "beginner",
                "also_known_as": "Bid-Ask Spread"
            },
            "Leverage": {
                "definition": "The use of borrowed capital to increase potential returns, expressed as a ratio (e.g., 100:1).",
                "example": "With 100:1 leverage, you can control $100,000 with $1,000 of your own money.",
                "usage": "Amplifies both profits and losses, allowing traders to control larger positions with less capital.",
                "difficulty": "beginner",
                "also_known_as": "Margin Trading"
            },
            "Currency Pair": {
                "definition": "Two currencies quoted against each other, showing how much of the quote currency is needed to buy one unit of the base currency.",
                "example": "In EUR/USD, EUR is the base currency and USD is the quote currency.",
                "usage": "All forex trading involves buying one currency and selling another simultaneously.",
                "difficulty": "beginner",
                "also_known_as": "FX Pair"
            }
        },
        
        "Technical Analysis": {
            "Moving Average": {
                "definition": "A trend-following indicator that smooths price data by creating a constantly updated average price.",
                "example": "20-day MA of EUR/USD averages the closing prices of the last 20 days.",
                "usage": "Identifies trend direction and potential support/resistance levels.",
                "difficulty": "beginner",
                "also_known_as": "MA"
            },
            "RSI": {
                "definition": "Relative Strength Index - momentum oscillator measuring speed and magnitude of price changes (0-100 scale).",
                "example": "RSI above 70 suggests overbought conditions, below 30 suggests oversold.",
                "usage": "Identifies potential reversal points and overbought/oversold conditions.",
                "difficulty": "intermediate",
                "also_known_as": "Relative Strength Index"
            },
            "MACD": {
                "definition": "Moving Average Convergence Divergence - trend-following momentum indicator using two moving averages.",
                "example": "MACD line crosses above signal line = potential bullish signal.",
                "usage": "Identifies trend changes, momentum shifts, and generates buy/sell signals.",
                "difficulty": "intermediate",
                "also_known_as": "Moving Average Convergence Divergence"
            },
            "Bollinger Bands": {
                "definition": "Volatility indicator consisting of a moving average with upper and lower bands based on standard deviation.",
                "example": "Price touching upper band may indicate overbought, lower band oversold.",
                "usage": "Measures volatility and identifies potential support/resistance levels.",
                "difficulty": "intermediate",
                "also_known_as": "BB"
            },
            "Support": {
                "definition": "A price level where downward price movement tends to stop due to concentration of buying interest.",
                "example": "EUR/USD repeatedly bounces off 1.1000 level, making it a support level.",
                "usage": "Used to identify potential buying opportunities and set stop losses.",
                "difficulty": "beginner",
                "also_known_as": "Support Level, Floor"
            },
            "Resistance": {
                "definition": "A price level where upward price movement tends to stop due to concentration of selling interest.",
                "example": "EUR/USD repeatedly fails to break above 1.1200, making it a resistance level.",
                "usage": "Used to identify potential selling opportunities and profit targets.",
                "difficulty": "beginner",
                "also_known_as": "Resistance Level, Ceiling"
            }
        },
        
        "Order Types": {
            "Market Order": {
                "definition": "An order to buy or sell immediately at the best available current market price.",
                "example": "Buying EUR/USD at current ask price of 1.1052 without waiting.",
                "usage": "For immediate execution when timing is more important than exact price.",
                "difficulty": "beginner",
                "also_known_as": "Instant Order"
            },
            "Limit Order": {
                "definition": "An order to buy or sell at a specific price or better, executed only when price reaches that level.",
                "example": "Buy EUR/USD at 1.1000 when current price is 1.1050 (buy limit).",
                "usage": "To enter positions at better prices or take profits at target levels.",
                "difficulty": "beginner",
                "also_known_as": "Pending Order"
            },
            "Stop Loss": {
                "definition": "An order that closes a position when price moves against you to limit losses.",
                "example": "Long EUR/USD at 1.1050 with stop loss at 1.1020 (30 pip risk).",
                "usage": "Essential risk management tool to limit downside on trades.",
                "difficulty": "beginner",
                "also_known_as": "SL, Stop Order"
            },
            "Take Profit": {
                "definition": "An order that closes a position when price moves in your favor to secure profits.",
                "example": "Long EUR/USD at 1.1050 with take profit at 1.1100 (50 pip target).",
                "usage": "Automatically secures profits without constant market monitoring.",
                "difficulty": "beginner",
                "also_known_as": "TP, Profit Target"
            }
        },
        
        "Risk Management": {
            "Position Size": {
                "definition": "The amount of currency units or lots traded in a single position.",
                "example": "Trading 0.5 lots (50,000 units) instead of 1 lot to reduce risk.",
                "usage": "Critical for risk management and capital preservation.",
                "difficulty": "beginner",
                "also_known_as": "Trade Size, Lot Size"
            },
            "Risk-Reward Ratio": {
                "definition": "The ratio comparing potential loss (risk) to potential profit (reward) on a trade.",
                "example": "Risk 30 pips for 60 pip target = 1:2 risk-reward ratio.",
                "usage": "Helps ensure profitable trading even with lower win rates.",
                "difficulty": "beginner",
                "also_known_as": "R:R, RRR"
            },
            "Drawdown": {
                "definition": "The peak-to-trough decline in account value during a losing period.",
                "example": "Account drops from $10,000 to $8,500 = 15% drawdown.",
                "usage": "Measures the largest loss from a peak, important for risk assessment.",
                "difficulty": "intermediate",
                "also_known_as": "Maximum Drawdown, DD"
            }
        },
        
        "Market Structure": {
            "Trend": {
                "definition": "The general direction of price movement over time - upward (bullish), downward (bearish), or sideways.",
                "example": "EUR/USD in uptrend: series of higher highs and higher lows over weeks.",
                "usage": "Fundamental concept for trend-following and mean-reversion strategies.",
                "difficulty": "beginner",
                "also_known_as": "Market Direction"
            },
            "Volatility": {
                "definition": "The degree of price variation over time, indicating market uncertainty or stability.",
                "example": "High volatility = large price swings, low volatility = small price movements.",
                "usage": "Affects strategy selection, position sizing, and stop loss placement.",
                "difficulty": "beginner",
                "also_known_as": "Market Volatility, Price Volatility"
            },
            "Liquidity": {
                "definition": "The ease with which an asset can be bought or sold without significantly affecting its price.",
                "example": "Major pairs like EUR/USD have high liquidity, exotic pairs have low liquidity.",
                "usage": "Affects spread costs, slippage, and ease of order execution.",
                "difficulty": "intermediate",
                "also_known_as": "Market Liquidity"
            }
        },
        
        "Advanced Concepts": {
            "Correlation": {
                "definition": "Statistical measure of how two currency pairs move in relation to each other.",
                "example": "EUR/USD and GBP/USD often have positive correlation (move in same direction).",
                "usage": "Important for portfolio diversification and risk management.",
                "difficulty": "advanced",
                "also_known_as": "Currency Correlation"
            },
            "Carry Trade": {
                "definition": "Strategy involving borrowing low-interest currency to buy high-interest currency to profit from rate differential.",
                "example": "Borrow JPY (low rate) to buy AUD (high rate) and earn interest differential.",
                "usage": "Long-term strategy that profits from interest rate differences between countries.",
                "difficulty": "advanced",
                "also_known_as": "Interest Rate Arbitrage"
            },
            "Arbitrage": {
                "definition": "Simultaneously buying and selling identical assets in different markets to profit from price differences.",
                "example": "EUR/USD cheaper on one broker than another, buy cheap and sell expensive simultaneously.",
                "usage": "Risk-free profit strategy, though opportunities are rare and short-lived in modern markets.",
                "difficulty": "advanced",
                "also_known_as": "Price Arbitrage"
            }
        }
    }

    # Filter terms function
    def filter_terms(terms, search_term, category_filter, difficulty_filter):
        filtered_terms = {}
        
        for category, term_list in terms.items():
            if category_filter != "All Categories" and category != category_filter:
                continue
                
            filtered_category = {}
            for term_name, term_data in term_list.items():
                if difficulty_filter != "All Levels" and term_data['difficulty'] != difficulty_filter.lower():
                    continue
                    
                search_text = f"{term_name} {term_data['definition']} {term_data['usage']} {term_data.get('also_known_as', '')}".lower()
                if not search_term or search_term.lower() in search_text:
                    filtered_category[term_name] = term_data
            
            if filtered_category:
                filtered_terms[category] = filtered_category
        
        return filtered_terms

    # Apply term filters
    filtered_terms = filter_terms(terminology, term_search, term_category, term_difficulty)

    # Display terms
    if not filtered_terms:
        st.warning("🔍 No terms match your search criteria. Try a different search term or filter.")
    else:
        for category, term_list in filtered_terms.items():
            st.markdown(f'<h2 class="category-header">{category}</h2>', unsafe_allow_html=True)
            
            for term_name, term_data in term_list.items():
                difficulty_class = f"difficulty-{term_data['difficulty']}"
                
                st.markdown('<div class="reference-section">', unsafe_allow_html=True)
                st.markdown(f'<div class="ref-card {difficulty_class}">', unsafe_allow_html=True)
                
                st.markdown(f'<h3 class="ref-name">{term_name}</h3>', unsafe_allow_html=True)
                st.markdown(f"**Definition:** {term_data['definition']}")
                
                st.markdown(f'<div class="ref-example"><strong>💡 Example:</strong><br>{term_data["example"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="ref-range"><strong>📈 Usage:</strong><br>{term_data["usage"]}</div>', unsafe_allow_html=True)
                
                if term_data.get('also_known_as'):
                    st.markdown(f"**Also known as:** {term_data['also_known_as']}")
                
                difficulty_emoji = {"beginner": "🟢", "intermediate": "🟡", "advanced": "🔴"}
                st.markdown(f"**Level:** {difficulty_emoji[term_data['difficulty']]} {term_data['difficulty'].title()}")
                
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

# Quick Reference Footer
st.markdown("---")
st.markdown('<div class="reference-section">', unsafe_allow_html=True)
st.markdown("### 🚀 **Quick Reference**")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📊 **Essential KPIs**")
    essential_kpis = [
        "📊 **Win Rate** - Track success percentage",
        "💰 **Total P&L** - Monitor profitability",
        "📉 **Maximum Drawdown** - Control risk",
        "⚖️ **Profit Factor** - Ensure positive expectancy"
    ]
    for kpi in essential_kpis:
        st.write(kpi)

with col2:
    st.markdown("#### 📚 **Basic Terms**")
    basic_terms = [
        "📍 **Pip** - Smallest price movement",
        "⚖️ **Leverage** - Borrowed capital for trading",
        "🛡️ **Stop Loss** - Risk management order",
        "🎯 **Support/Resistance** - Key price levels"
    ]
    for term in basic_terms:
        st.write(term)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("*This comprehensive reference guide combines essential KPIs and trading terminology to support your trading education and performance analysis.*")