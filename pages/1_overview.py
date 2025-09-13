"""
Overview Page - Clean and simple trading signals dashboard
"""
import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import sys
from pathlib import Path

st.set_page_config(page_title="Overview", page_icon="📈", layout="wide")

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from utils.auth import require_authentication, render_user_info
    from pages.components.signal_table import render_signal_table, get_signal_status
    # Require authentication for this page
    user_info = require_authentication()
    render_user_info()
    imports_successful = True
except ImportError as e:
    st.warning("⚠️ Authentication modules not found - running in demo mode")
    user_info = {"username": "demo", "role": "admin"}
    imports_successful = False
    from pages.components.signal_table import render_signal_table, get_signal_status

# Clean, simple CSS styling
st.markdown("""
<style>
    /* Clean title styling */
    .dashboard-title {
        font-size: 2.5rem;
        font-weight: 600;
        text-align: center;
        color: #2c3e50;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 3px solid #3498db;
    }
    
    /* Simple metric cards */
    [data-testid="metric-container"] {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Clean section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3498db;
    }
    
    /* Simple table styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #e9ecef;
    }
    
    /* Clean button styling */
    .stButton > button {
        background: #3498db;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: background-color 0.2s;
    }
    
    .stButton > button:hover {
        background: #2980b9;
    }
    
    /* Status indicators */
    .status-good { color: #27ae60; font-weight: 600; }
    .status-warning { color: #f39c12; font-weight: 600; }
    .status-bad { color: #e74c3c; font-weight: 600; }
    
    /* Clean info boxes */
    .info-box {
        background: #f8f9fa;
        border-left: 4px solid #3498db;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="dashboard-title">📈 Trading Signals Dashboard</h1>', unsafe_allow_html=True)

# Helper function for API calls
def call_api(endpoint, method="GET", data=None):
    """Call backend API with fallback to demo data"""
    try:
        base_url = "http://0.0.0.0:8000"
        url = f"{base_url}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise requests.exceptions.ConnectionError("API not available")
            
    except requests.exceptions.RequestException:
        st.info("🔄 Running in demo mode - using sample data")
        return get_demo_data(endpoint)

def is_forex_market_open():
    """Simple market hours check"""
    now = datetime.utcnow()
    weekday = now.weekday()  # Monday = 0, Sunday = 6
    
    # Market closed on weekends
    if weekday >= 5:  # Saturday = 5, Sunday = 6
        return False
    elif weekday == 4:  # Friday - close at 22:00 UTC
        return now.hour < 22
    else:
        return True

def get_demo_data(endpoint):
    """Provide simple demo data with FIXED signal structure matching Signal model"""
    import random
    
    if "/api/signals/recent" in endpoint:
        if not is_forex_market_open():
            return []
        
        # Demo signals with proper structure matching Signal model
        signals = []
        symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
        strategies = ["ema_rsi", "donchian_atr", "meanrev_bb", "macd_strategy"]
        current_time = datetime.now()
        
        for i in range(8):  # Fewer signals for cleaner view
            signal_time = current_time - timedelta(minutes=random.randint(5, 60))
            expires_time = signal_time + timedelta(hours=random.randint(1, 4))
            
            signals.append({
                "id": 100 + i,
                "symbol": random.choice(symbols),
                "action": random.choice(["BUY", "SELL"]),
                "price": round(random.uniform(1.0, 150.0), 5),
                "sl": round(random.uniform(1.0, 150.0), 5),
                "tp": round(random.uniform(1.0, 150.0), 5),
                "confidence": random.uniform(0.65, 0.95),
                "strategy": random.choice(strategies),
                "issued_at": signal_time.isoformat() + "Z",
                "expires_at": expires_time.isoformat() + "Z",
                "result": random.choice(["PENDING", "PENDING", "PENDING", "WIN", "LOSS", "EXPIRED"]),
                "sent_to_whatsapp": random.choice([True, False, True]),
                "blocked_by_risk": random.choice([False, False, False, True])
            })
        
        return signals
    
    elif "/api/risk/status" in endpoint:
        return {
            "kill_switch_enabled": False,
            "daily_loss_limit": 1000,
            "current_daily_loss": random.randint(50, 200),
            "signals_today": random.randint(5, 15)
        }
    
    elif "/api/signals/stats" in endpoint:
        return {
            "success_rate": random.randint(65, 85),
            "total_signals_today": random.randint(8, 20),
            "active_signals": random.randint(2, 6),
            "buy_signals_today": random.randint(3, 12),
            "sell_signals_today": random.randint(3, 12)
        }
    
    return None

# Load data
@st.cache_data(ttl=30)
def load_dashboard_data():
    """Load all dashboard data"""
    return {
        "signals": call_api("/api/signals/recent?limit=10"),
        "risk_status": call_api("/api/risk/status"),
        "stats": call_api("/api/signals/stats")
    }

# Auto-refresh setup
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = 0

current_time = datetime.now().timestamp()
if current_time - st.session_state.last_refresh > 30:  # 30 second refresh
    st.cache_data.clear()
    st.session_state.last_refresh = current_time

# Load dashboard data
data = load_dashboard_data()
signals = data.get("signals", [])
risk_status = data.get("risk_status", {})
stats = data.get("stats", {})

# Quick Status Overview
st.markdown('<div class="section-header">System Status</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    kill_switch = risk_status.get('kill_switch_enabled', False)
    status = "🔴 STOPPED" if kill_switch else "🟢 ACTIVE"
    st.metric("Trading Status", status)

with col2:
    market_open = is_forex_market_open()
    market_status = "🟢 OPEN" if market_open else "🔴 CLOSED"
    st.metric("Market Status", market_status)

with col3:
    # Use FIXED metrics calculation from actual signals data
    if signals:
        active_count = sum(1 for s in signals if get_signal_status(s)[0].startswith('🟢'))
        st.metric("Active Signals", active_count)
    else:
        today_signals = stats.get('total_signals_today', 0)
        st.metric("Today's Signals", today_signals)

with col4:
    success_rate = stats.get('success_rate', 0)
    st.metric("Success Rate", f"{success_rate}%")

st.markdown("---")

# Separate signals by type
def separate_signals_by_type(signals):
    """Separate signals into Forex Major, Crypto, and Metals & Oil categories"""
    forex_majors = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD', 
                   'EURJPY', 'GBPJPY', 'EURGBP', 'AUDJPY', 'EURAUD', 'EURCAD', 'EURCHF', 'AUDCAD']
    crypto_pairs = ['BTCUSD', 'ETHUSD', 'BTCEUR', 'ETHEUR', 'BTCGBP', 'ETHGBP', 'LTCUSD', 'ADAUSD', 'DOGUSD', 'SOLUSD', 'AVAXUSD']
    metals_oil = ['XAUUSD', 'XAGUSD', 'USOIL', 'UKOUSD', 'XPTUSD', 'XPDUSD', 'WTIUSD', 'XBRUSD', 'XPDUSD', 'XRHHUSD']
    
    forex_signals = [s for s in signals if s.get('symbol', '').upper() in forex_majors]
    crypto_signals = [s for s in signals if s.get('symbol', '').upper() in crypto_pairs]
    metals_oil_signals = [s for s in signals if s.get('symbol', '').upper() in metals_oil]
    
    return forex_signals, crypto_signals, metals_oil_signals

# Main signals sections
st.markdown('<div class="section-header">📊 Live Trading Signals</div>', unsafe_allow_html=True)

if signals and len(signals) > 0:
    # Separate signals by type
    forex_signals, crypto_signals, metals_oil_signals = separate_signals_by_type(signals)
    
    # Create tabs for Forex Major, Crypto, and Metals & Oil
    tab1, tab2, tab3 = st.tabs(["💱 Forex Major", "₿ Crypto", "🥇 Metals & Oil"])
    
    with tab1:
        st.markdown('<h4>Major Currency Pairs</h4>', unsafe_allow_html=True)
        if forex_signals:
            render_signal_table(forex_signals, title="", show_details=False, max_rows=10)
            
            # Forex metrics
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            
            with col1:
                forex_active = sum(1 for s in forex_signals if get_signal_status(s)[0].startswith('🟢'))
                st.metric("Active Forex", forex_active)
            
            with col2:
                forex_buy = sum(1 for s in forex_signals if s.get('action') == 'BUY')
                st.metric("Forex Buy", forex_buy)
            
            with col3:
                forex_sell = sum(1 for s in forex_signals if s.get('action') == 'SELL')
                st.metric("Forex Sell", forex_sell)
            
            with col4:
                forex_confidence = sum(s.get('confidence', 0) for s in forex_signals) / len(forex_signals) if forex_signals else 0
                st.metric("Avg Confidence", f"{forex_confidence:.1%}")
        else:
            st.info("📊 No active Forex major currency signals. The system is monitoring EUR/USD, GBP/USD, USD/JPY and other major pairs.")
    
    with tab2:
        st.markdown('<h4>Cryptocurrency Pairs</h4>', unsafe_allow_html=True)
        if crypto_signals:
            render_signal_table(crypto_signals, title="", show_details=False, max_rows=10)
            
            # Crypto metrics
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            
            with col1:
                crypto_active = sum(1 for s in crypto_signals if get_signal_status(s)[0].startswith('🟢'))
                st.metric("Active Crypto", crypto_active)
            
            with col2:
                crypto_buy = sum(1 for s in crypto_signals if s.get('action') == 'BUY')
                st.metric("Crypto Buy", crypto_buy)
            
            with col3:
                crypto_sell = sum(1 for s in crypto_signals if s.get('action') == 'SELL')
                st.metric("Crypto Sell", crypto_sell)
            
            with col4:
                crypto_confidence = sum(s.get('confidence', 0) for s in crypto_signals) / len(crypto_signals) if crypto_signals else 0
                st.metric("Avg Confidence", f"{crypto_confidence:.1%}")
        else:
            st.info("₿ No active cryptocurrency signals. The system is monitoring BTC/USD and ETH/USD pairs 24/7.")
    
    with tab3:
        st.markdown('<h4>Metals & Oil Markets</h4>', unsafe_allow_html=True)
        if metals_oil_signals:
            render_signal_table(metals_oil_signals, title="", show_details=False, max_rows=10)
            
            # Metals & Oil metrics
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            
            with col1:
                metals_active = sum(1 for s in metals_oil_signals if get_signal_status(s)[0].startswith('🟢'))
                st.metric("Active Metals/Oil", metals_active)
            
            with col2:
                metals_buy = sum(1 for s in metals_oil_signals if s.get('action') == 'BUY')
                st.metric("Metals/Oil Buy", metals_buy)
            
            with col3:
                metals_sell = sum(1 for s in metals_oil_signals if s.get('action') == 'SELL')
                st.metric("Metals/Oil Sell", metals_sell)
            
            with col4:
                metals_confidence = sum(s.get('confidence', 0) for s in metals_oil_signals) / len(metals_oil_signals) if metals_oil_signals else 0
                st.metric("Avg Confidence", f"{metals_confidence:.1%}")
        else:
            st.info("🥇 No active metals & oil signals. The system is monitoring Gold (XAU/USD), Silver (XAG/USD), Oil (US/UK), and precious metals markets 24/7.")
    
    # Global action buttons
    st.markdown("---")
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    with col2:
        # Total active signals
        total_active = sum(1 for s in signals if get_signal_status(s)[0].startswith('🟢'))
        st.metric("Total Active", total_active)
    
    with col3:
        total_buy = sum(1 for s in signals if s.get('action') == 'BUY')
        st.metric("Total Buy", total_buy)
    
    with col4:
        total_sell = sum(1 for s in signals if s.get('action') == 'SELL')
        st.metric("Total Sell", total_sell)

else:
    # Market closed or no signals message
    if not is_forex_market_open():
        st.markdown("""
        <div class="info-box">
            <h4>🌙 Market is Currently Closed</h4>
            <p>The Forex market is closed on weekends. Trading signals will resume when the market opens.</p>
            <p><strong>Market Hours:</strong> Sunday 22:00 UTC - Friday 22:00 UTC</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
            <h4>📊 No Active Signals</h4>
            <p>No trading signals are currently available. The system is monitoring the market and will generate signals when conditions are met.</p>
        </div>
        """, unsafe_allow_html=True)
    
    if st.button("🔄 Check for New Signals", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Risk Management Summary
st.markdown("---")
st.markdown('<div class="section-header">🛡️ Risk Management</div>', unsafe_allow_html=True)

if risk_status:
    col1, col2 = st.columns(2)
    
    with col1:
        daily_loss = risk_status.get('current_daily_loss', 0)
        loss_limit = risk_status.get('daily_loss_limit', 1000)
        remaining = loss_limit - daily_loss
        
        st.metric("Daily Loss Limit", f"${loss_limit:.0f}")
        st.metric("Remaining Budget", f"${remaining:.0f}")
        
        # Simple progress bar
        if loss_limit > 0:
            progress = min(daily_loss / loss_limit, 1.0)
            st.progress(progress)
            st.caption(f"Used: ${daily_loss:.0f} of ${loss_limit:.0f}")
    
    with col2:
        st.metric("Kill Switch", "🔴 ON" if risk_status.get('kill_switch_enabled') else "🟢 OFF")
        
        # Simple risk status indicator
        if daily_loss / loss_limit > 0.8:
            st.error("⚠️ High risk exposure - approaching daily limit")
        elif daily_loss / loss_limit > 0.6:
            st.warning("⚠️ Moderate risk exposure")
        else:
            st.success("✅ Low risk exposure")

# Quick Navigation
st.markdown("---")
st.markdown('<div class="section-header">⚡ Quick Actions</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("⚙️ Configure Strategies", use_container_width=True):
        st.switch_page("pages/2_strategies.py")

with col2:
    if st.button("🛡️ Risk Settings", use_container_width=True):
        st.switch_page("pages/3_risk.py")

with col3:
    if st.button("📰 Market News", use_container_width=True):
        st.switch_page("pages/7_news.py")

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #7f8c8d; font-size: 0.9rem;">
    Last updated: {datetime.now().strftime('%H:%M:%S')} | Auto-refresh every 30 seconds
</div>
""", unsafe_allow_html=True)