"""
Terminology Page - Comprehensive Forex Trading Glossary
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

st.set_page_config(page_title="Terminology", page_icon="📚", layout="wide")

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

# Professional Terminology Page Styling
st.markdown("""
<style>
    .terminology-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.8rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .terminology-section {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
    }
    
    .term-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 1.8rem;
        border-radius: 15px;
        border: 2px solid #e1e8ed;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        transition: transform 0.2s ease;
    }
    
    .term-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.15);
    }
    
    .term-name {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.4rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .term-example {
        background: linear-gradient(135deg, #a8e6cf 0%, #88d8c0 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #2d3436;
        border-left: 4px solid #00b894;
    }
    
    .term-usage {
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
    
    .difficulty-beginner {
        border-left: 5px solid #00b894;
        background: linear-gradient(135deg, #a8e6cf 0%, #88d8c0 100%);
    }
    
    .difficulty-intermediate {
        border-left: 5px solid #f39c12;
        background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);
    }
    
    .difficulty-advanced {
        border-left: 5px solid #e74c3c;
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
    }
    
    .alphabetical-index {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
    }
    
    .alphabetical-index a {
        color: white;
        text-decoration: none;
        margin: 0 0.5rem;
        font-weight: bold;
        font-size: 1.1rem;
    }
    
    .alphabetical-index a:hover {
        color: #fdcb6e;
    }
</style>
""", unsafe_allow_html=True)

# Page Title
st.markdown('<h1 class="terminology-title">📚 Forex Trading Terminology</h1>', unsafe_allow_html=True)
st.markdown("### *Comprehensive Glossary for Professional Trading*")
st.markdown("---")

# Search functionality
st.markdown('<div class="search-box">', unsafe_allow_html=True)
st.markdown("### 🔍 **Find Trading Terms**")
search_term = st.text_input("Search for trading terms:", placeholder="e.g., Pip, Leverage, Moving Average, RSI...")

col1, col2 = st.columns(2)
with col1:
    category_filter = st.selectbox(
        "Filter by category:",
        ["All Categories", "Basic Forex", "Technical Analysis", "Fundamental Analysis", "Risk Management", "Order Types", "Market Structure", "Advanced Concepts"]
    )

with col2:
    difficulty_filter = st.selectbox(
        "Filter by difficulty:",
        ["All Levels", "Beginner", "Intermediate", "Advanced"]
    )

st.markdown('</div>', unsafe_allow_html=True)

# Comprehensive terminology database
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
        "Currency Pair": {
            "definition": "Two currencies quoted against each other, showing how much of the quote currency is needed to buy one unit of the base currency.",
            "example": "In EUR/USD, EUR is the base currency and USD is the quote currency.",
            "usage": "All forex trading involves buying one currency and selling another simultaneously.",
            "difficulty": "beginner",
            "also_known_as": "FX Pair"
        },
        "Leverage": {
            "definition": "The use of borrowed capital to increase potential returns, expressed as a ratio (e.g., 100:1).",
            "example": "With 100:1 leverage, you can control $100,000 with $1,000 of your own money.",
            "usage": "Amplifies both profits and losses, allowing traders to control larger positions with less capital.",
            "difficulty": "beginner",
            "also_known_as": "Margin Trading"
        },
        "Margin": {
            "definition": "The required deposit to open a leveraged position, expressed as a percentage of the full position size.",
            "example": "2% margin means you need $2,000 to control a $100,000 position.",
            "usage": "Acts as collateral for leveraged trades and determines position sizes.",
            "difficulty": "beginner",
            "also_known_as": "Required Margin, Initial Margin"
        },
        "Lot": {
            "definition": "Standard unit size for trading currencies. Standard lot = 100,000 units of base currency.",
            "example": "1 standard lot of EUR/USD = 100,000 euros. Mini lot = 10,000, Micro lot = 1,000.",
            "usage": "Determines position size and risk per pip movement.",
            "difficulty": "beginner",
            "also_known_as": "Contract Size"
        },
        "Ask Price": {
            "definition": "The price at which you can buy a currency pair (the broker's selling price).",
            "example": "If EUR/USD ask is 1.1052, you pay 1.1052 USD for each EUR.",
            "usage": "Always higher than bid price, used when opening long positions.",
            "difficulty": "beginner",
            "also_known_as": "Offer Price"
        },
        "Bid Price": {
            "definition": "The price at which you can sell a currency pair (the broker's buying price).",
            "example": "If EUR/USD bid is 1.1050, you receive 1.1050 USD for each EUR sold.",
            "usage": "Always lower than ask price, used when opening short positions.",
            "difficulty": "beginner",
            "also_known_as": "Selling Price"
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
        "EMA": {
            "definition": "Exponential Moving Average - gives more weight to recent prices than older prices.",
            "example": "12-period EMA reacts faster to price changes than 12-period SMA.",
            "usage": "Better for identifying trend changes early, commonly used in crossover strategies.",
            "difficulty": "intermediate",
            "also_known_as": "Exponential Moving Average"
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
        },
        "Candlestick": {
            "definition": "A charting method showing open, high, low, and close prices for a specific time period.",
            "example": "A green/white candle shows close > open (bullish), red/black shows close < open (bearish).",
            "usage": "Provides visual representation of price action and market sentiment.",
            "difficulty": "beginner",
            "also_known_as": "Candle, Japanese Candlestick"
        },
        "ATR": {
            "definition": "Average True Range - measures market volatility by calculating average range of price movements.",
            "example": "Higher ATR indicates more volatile market conditions.",
            "usage": "Used for position sizing, stop loss placement, and volatility filtering.",
            "difficulty": "intermediate",
            "also_known_as": "Average True Range"
        },
        "Fibonacci Retracement": {
            "definition": "Technical analysis tool using horizontal lines at 23.6%, 38.2%, 50%, 61.8%, and 78.6% levels.",
            "example": "After uptrend, price may retrace to 61.8% Fibonacci level before continuing up.",
            "usage": "Identifies potential support/resistance levels and entry/exit points.",
            "difficulty": "intermediate",
            "also_known_as": "Fib Levels, Fibonacci"
        },
        "Stochastic": {
            "definition": "Momentum indicator comparing closing price to price range over specific period (0-100 scale).",
            "example": "Stochastic above 80 = overbought, below 20 = oversold conditions.",
            "usage": "Identifies overbought/oversold conditions and potential reversal points.",
            "difficulty": "intermediate",
            "also_known_as": "Stochastic Oscillator"
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
        },
        "Stop Order": {
            "definition": "An order to buy above current market price or sell below current market price.",
            "example": "Buy stop at 1.1100 when price is 1.1050 (breakout strategy).",
            "usage": "Used for breakout strategies and trend continuation trades.",
            "difficulty": "intermediate",
            "also_known_as": "Stop Entry"
        },
        "Trailing Stop": {
            "definition": "A stop loss that automatically adjusts in favorable direction as price moves in profit.",
            "example": "50-pip trailing stop follows price up but stays fixed when price moves down.",
            "usage": "Protects profits while allowing for continued favorable price movement.",
            "difficulty": "intermediate",
            "also_known_as": "Dynamic Stop"
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
        },
        "Margin Call": {
            "definition": "A broker's demand for additional funds when account equity falls below required margin level.",
            "example": "Account equity drops to 50% of required margin, triggering margin call.",
            "usage": "Warning sign of excessive risk and potential forced position closure.",
            "difficulty": "intermediate",
            "also_known_as": "Margin Alert"
        },
        "Stop Out": {
            "definition": "Automatic closure of positions when account equity falls below stop out level (usually 20-30%).",
            "example": "Broker automatically closes all positions when equity hits 20% of used margin.",
            "usage": "Final risk protection to prevent negative account balance.",
            "difficulty": "intermediate",
            "also_known_as": "Stop Out Level, Forced Liquidation"
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
        },
        "Slippage": {
            "definition": "The difference between expected price and actual execution price of an order.",
            "example": "Order to buy at 1.1050 executes at 1.1052 due to fast market movement.",
            "usage": "Common during high volatility periods and affects trading costs.",
            "difficulty": "intermediate",
            "also_known_as": "Price Slippage"
        },
        "Gap": {
            "definition": "A break between prices where no trading occurs, often seen at market open after weekend.",
            "example": "EUR/USD closes Friday at 1.1050, opens Monday at 1.1080 (30-pip gap).",
            "usage": "Can provide trading opportunities or cause unexpected losses.",
            "difficulty": "intermediate",
            "also_known_as": "Price Gap"
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
        "Swap": {
            "definition": "The interest rate differential paid or received for holding a position overnight.",
            "example": "Long AUD/JPY typically receives positive swap due to interest rate differential.",
            "usage": "Affects profitability of positions held overnight, especially for carry trades.",
            "difficulty": "advanced",
            "also_known_as": "Rollover, Overnight Interest"
        },
        "Hedge": {
            "definition": "Taking opposite position to reduce risk exposure on existing trade or portfolio.",
            "example": "Long EUR/USD and short EUR/GBP to reduce EUR exposure while maintaining USD/GBP view.",
            "usage": "Risk management technique to reduce overall portfolio volatility.",
            "difficulty": "advanced",
            "also_known_as": "Hedging"
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

# Filter terms based on search and filters
def filter_terms(terms, search_term, category_filter, difficulty_filter):
    filtered_terms = {}
    
    for category, term_list in terms.items():
        if category_filter != "All Categories" and category != category_filter:
            continue
            
        filtered_category = {}
        for term_name, term_data in term_list.items():
            # Filter by difficulty
            if difficulty_filter != "All Levels" and term_data['difficulty'] != difficulty_filter.lower():
                continue
                
            # Search in term name, definition, and usage
            search_text = f"{term_name} {term_data['definition']} {term_data['usage']} {term_data.get('also_known_as', '')}".lower()
            if not search_term or search_term.lower() in search_text:
                filtered_category[term_name] = term_data
        
        if filtered_category:
            filtered_terms[category] = filtered_category
    
    return filtered_terms

# Apply filters
filtered_terms = filter_terms(terminology, search_term, category_filter, difficulty_filter)

# Display summary statistics
st.markdown('<div class="terminology-section">', unsafe_allow_html=True)
st.markdown("### 📈 **Glossary Overview**")

col1, col2, col3, col4 = st.columns(4)

total_terms = sum(len(term_list) for term_list in terminology.values())
filtered_count = sum(len(term_list) for term_list in filtered_terms.values())
beginner_terms = sum(1 for category in terminology.values() 
                    for term in category.values() if term['difficulty'] == 'beginner')

with col1:
    st.metric("Total Terms", f"{total_terms}", delta="Comprehensive coverage")

with col2:
    st.metric("Showing", f"{filtered_count}", delta=f"Filtered from {total_terms}")

with col3:
    st.metric("Beginner Terms", f"{beginner_terms}", delta="Easy to understand")

with col4:
    st.metric("Categories", f"{len(terminology)}", delta="Complete coverage")

st.markdown('</div>', unsafe_allow_html=True)

# Alphabetical index for quick navigation
st.markdown('<div class="terminology-section">', unsafe_allow_html=True)
st.markdown("### 🔤 **Quick Navigation**")
st.markdown('<div class="alphabetical-index">', unsafe_allow_html=True)

# Create alphabetical index
all_terms = []
for category in terminology.values():
    all_terms.extend(category.keys())

# Group by first letter
letters = sorted(set(term[0].upper() for term in all_terms))
for letter in letters:
    if st.button(letter, key=f"letter_{letter}"):
        st.session_state.selected_letter = letter

alphabet_links = " | ".join([f'<a href="#{letter}">{letter}</a>' for letter in letters])
st.markdown(f'{alphabet_links}', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Display filtered terms
if not filtered_terms:
    st.warning("🔍 No terms match your search criteria. Try a different search term or filter.")
else:
    for category, term_list in filtered_terms.items():
        st.markdown(f'<h2 class="category-header">{category}</h2>', unsafe_allow_html=True)
        
        # Sort terms alphabetically
        sorted_terms = sorted(term_list.items())
        
        for term_name, term_data in sorted_terms:
            # Determine card styling based on difficulty
            difficulty_class = f"difficulty-{term_data['difficulty']}"
            
            st.markdown('<div class="terminology-section">', unsafe_allow_html=True)
            st.markdown(f'<div class="term-card {difficulty_class}">', unsafe_allow_html=True)
            
            # Term Name and Definition
            st.markdown(f'<h3 class="term-name" id="{term_name[0].upper()}">{term_name}</h3>', unsafe_allow_html=True)
            st.markdown(f"**Definition:** {term_data['definition']}")
            
            # Also known as
            if 'also_known_as' in term_data:
                st.markdown(f"**Also known as:** {term_data['also_known_as']}")
            
            # Example
            st.markdown(f'<div class="term-example"><strong>💡 Example:</strong><br>{term_data["example"]}</div>', unsafe_allow_html=True)
            
            # Usage
            st.markdown(f'<div class="term-usage"><strong>🎯 Usage:</strong><br>{term_data["usage"]}</div>', unsafe_allow_html=True)
            
            # Difficulty indicator
            difficulty_emoji = {"beginner": "🟢", "intermediate": "🟡", "advanced": "🔴"}
            st.markdown(f"**Level:** {difficulty_emoji[term_data['difficulty']]} {term_data['difficulty'].title()}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# Learning path recommendations
st.markdown("---")
st.markdown('<div class="terminology-section">', unsafe_allow_html=True)
st.markdown("### 🎓 **Recommended Learning Path**")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 🟢 **Start Here (Beginner)**")
    beginner_terms = [
        "📊 **Pip** - Basic price measurement",
        "💱 **Currency Pair** - Foundation of forex",
        "📈 **Spread** - Trading cost understanding",
        "💰 **Leverage** - Risk amplification tool",
        "🔄 **Market Order** - Basic order type",
        "🛡️ **Stop Loss** - Risk protection",
        "📈 **Support/Resistance** - Key price levels"
    ]
    for term in beginner_terms:
        st.write(term)

with col2:
    st.markdown("#### 🟡 **Build Skills (Intermediate)**")
    intermediate_terms = [
        "📊 **RSI** - Momentum indicator",
        "📈 **Moving Averages** - Trend identification", 
        "🎯 **Risk-Reward Ratio** - Trade planning",
        "⚡ **ATR** - Volatility measurement",
        "📋 **Limit Orders** - Better entry/exit",
        "🔄 **MACD** - Trend momentum",
        "📊 **Bollinger Bands** - Volatility analysis"
    ]
    for term in intermediate_terms:
        st.write(term)

with col3:
    st.markdown("#### 🔴 **Master Advanced (Expert)**")
    advanced_terms = [
        "🔗 **Correlation** - Portfolio relationships",
        "💹 **Carry Trade** - Interest rate strategy",
        "🔄 **Hedge** - Risk reduction technique",
        "⚖️ **Arbitrage** - Risk-free profits",
        "🔄 **Swap** - Overnight interest",
        "📊 **Fibonacci** - Advanced retracements",
        "🎯 **Position Sizing** - Capital allocation"
    ]
    for term in advanced_terms:
        st.write(term)

# Study tips
st.markdown("---")
st.markdown("#### 💡 **Study Tips for Trading Terminology**")

tips = [
    "📚 **Start with Basics**: Master beginner terms before moving to advanced concepts",
    "🔄 **Practice Daily**: Use new terms in your trading journal to reinforce learning",
    "📊 **Visual Learning**: Draw charts and diagrams to understand concepts better",
    "💬 **Discuss with Others**: Join trading communities to use terminology in context",
    "📖 **Read Market News**: Financial news uses these terms regularly",
    "🎯 **Apply in Practice**: Use demo accounts to apply theoretical knowledge",
    "📝 **Create Flashcards**: Quick review method for memorizing definitions"
]

for tip in tips:
    st.write(tip)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("*This comprehensive glossary covers essential forex trading terminology. Master these terms to communicate effectively in professional trading environments and understand market analysis.*")