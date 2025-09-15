"""
Overview Page - Clean and simple trading signals dashboard
"""
import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, TypedDict, Mapping, Sequence

st.set_page_config(page_title="Overview", page_icon="📈", layout="wide")

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import config for deployment-safe API URLs
from config import get_backend_url

# Import timezone utilities
sys.path.append(str(project_root / "utils"))
from timezone_utils import format_saudi_time, to_saudi_time

# Type definitions for API responses
class SignalDTO(TypedDict, total=False):
    id: int
    symbol: str
    action: str
    price: float
    confidence: float
    strategy: str
    issued_at: str
    expires_at: str
    sl: Optional[float]
    tp: Optional[float]

class RiskStatusDTO(TypedDict, total=False):
    kill_switch: bool
    daily_loss_limit: float
    current_loss: float

class StatsDTO(TypedDict, total=False):
    total_signals: int
    active_signals: int
    win_rate: float

from pages.components.signal_table import render_signal_table, get_signal_status

# No authentication required
user_info = {"username": "user", "role": "admin"}

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

# Production mode check function
@st.cache_data(ttl=60)  # Cache for 1 minute
def get_production_mode_status() -> Dict[str, Any]:
    """Get production mode status from backend API"""
    try:
        base_url = get_backend_url()
        response = requests.get(f"{base_url}/api/system/production-mode", timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            # Fallback to demo mode if API fails
            return {
                "is_production_mode": False,
                "status": "🟡 DEMO MODE",
                "data_source": "demo",
                "note": "Production mode check failed"
            }
    except requests.exceptions.RequestException:
        # Fallback to demo mode if API not available
        return {
            "is_production_mode": False,
            "status": "🟡 DEMO MODE",
            "data_source": "demo",
            "note": "API not available"
        }

# Auto-refresh functionality
if "auto_refresh_enabled" not in st.session_state:
    st.session_state.auto_refresh_enabled = True
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

# Production mode status
production_status = get_production_mode_status()
is_production = production_status.get('is_production_mode', False)
status_text = production_status.get('status', '🟡 DEMO MODE')

# Production mode indicator at the top
if is_production:
    st.markdown(
        f'<div style="background: linear-gradient(90deg, #27ae60, #2ecc71); color: white; padding: 1rem; border-radius: 8px; text-align: center; font-weight: 600; font-size: 1.1rem; margin-bottom: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">'  
        f'🟢 LIVE MARKET DATA | Real-time trading signals powered by live market feeds'  
        f'</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        f'<div style="background: linear-gradient(90deg, #f39c12, #e67e22); color: white; padding: 1rem; border-radius: 8px; text-align: center; font-weight: 600; font-size: 1.1rem; margin-bottom: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">'  
        f'🟡 DEMO MODE | Displaying sample data for demonstration purposes'  
        f'</div>',
        unsafe_allow_html=True
    )

# Auto-refresh controls and timer
col_title, col_refresh = st.columns([4, 1])

with col_title:
    st.markdown('<h1 class="dashboard-title">📈 Trading Signals Dashboard</h1>', unsafe_allow_html=True)

with col_refresh:
    # Auto-refresh toggle
    auto_refresh = st.toggle("Auto-refresh (30s)", value=st.session_state.auto_refresh_enabled)
    st.session_state.auto_refresh_enabled = auto_refresh
    
    # Manual refresh button
    if st.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.session_state.last_refresh = time.time()
        st.rerun()

# Auto-refresh timer logic - use st.components for reliable auto-refresh  
if st.session_state.auto_refresh_enabled:
    import streamlit.components.v1 as components
    
    # Use Streamlit components for reliable auto-refresh JavaScript execution
    components.html(
        """
        <meta http-equiv="refresh" content="30">
        <script>
        // Live countdown timer that actually works in Streamlit
        let timeLeft = 30;
        let countdownElement = null;
        
        function updateCountdown() {
            if (!countdownElement) {
                countdownElement = document.createElement('div');
                countdownElement.style.cssText = 'position: fixed; top: 80px; right: 20px; background: rgba(0,0,0,0.8); color: white; padding: 8px 12px; border-radius: 6px; z-index: 9999; font-family: monospace; font-size: 12px;';
                document.body.appendChild(countdownElement);
            }
            
            timeLeft--;
            if (timeLeft <= 0) {
                countdownElement.innerHTML = '🔄 Refreshing...';
                // Force page reload
                setTimeout(() => window.location.reload(), 500);
                return;
            }
            
            countdownElement.innerHTML = '⏱️ Refresh: ' + timeLeft + 's';
        }
        
        // Start countdown immediately
        updateCountdown();
        const countdownTimer = setInterval(updateCountdown, 1000);
        
        // Ensure cleanup
        window.addEventListener('beforeunload', () => clearInterval(countdownTimer));
        </script>
        """,
        height=0,  # Invisible component, just for JavaScript execution
    )
else:
    # Show manual refresh reminder when auto-refresh is disabled
    st.markdown('<div style="text-align: right; color: #999; font-size: 0.9rem; margin-bottom: 1rem;">Auto-refresh disabled - use refresh button for latest data</div>', unsafe_allow_html=True)

# Helper function for API calls
def call_api(endpoint: str, method: str = "GET", data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any] | List[Dict[str, Any]]]:
    """Call backend API with fallback to demo data"""
    response = None  # Initialize response
    try:
        if method not in ["GET", "POST"]:
            raise ValueError(f"Unsupported method: {method}")
            
        base_url = get_backend_url()
        url = f"{base_url}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        
        if response and response.status_code == 200:
            result = response.json()
            # Validate JSON type
            if isinstance(result, (dict, list)):
                return result
            return None
        else:
            raise requests.exceptions.ConnectionError("API not available")
            
    except requests.exceptions.RequestException:
        st.info("🔄 Running in demo mode - using sample data")
        return get_demo_data(endpoint)

def is_forex_market_open():
    """Forex market hours check: Sunday 21:00 UTC - Friday 21:00 UTC"""
    now = datetime.utcnow()
    weekday = now.weekday()  # Monday = 0, Sunday = 6
    hour = now.hour
    
    # Saturday = Market CLOSED
    if weekday == 5:  # Saturday
        return False
    
    # Sunday = Market OPEN after 21:00 UTC
    elif weekday == 6:  # Sunday
        return hour >= 21
    
    # Monday to Thursday = Market OPEN
    elif weekday in [0, 1, 2, 3]:  # Monday-Thursday
        return True
    
    # Friday = Market OPEN until 21:00 UTC
    elif weekday == 4:  # Friday
        return hour < 21
    
    return False

def get_demo_data(endpoint):
    """Provide simple demo data with FIXED signal structure matching Signal model"""
    import random
    
    if "/api/signals/recent" in endpoint:
        # Always show signals - crypto trades 24/7 and users need historical data
        # Demo signals with proper structure matching Signal model
        signals = []
        # Use mix of symbols from all three categories (forex, crypto, metals/oil)
        forex_symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "EURGBP", "EURJPY", "GBPJPY"]
        crypto_symbols = ["BTCUSD", "ETHUSD", "ADAUSD", "DOGEUSD", "SOLUSD", "BNBUSD", "XRPUSD", "MATICUSD"] 
        metals_oil_symbols = ["XAUUSD", "XAGUSD", "XPTUSD", "XPDUSD", "USOIL", "UKOUSD", "WTIUSD", "XBRUSD"]
        symbols = forex_symbols + crypto_symbols + metals_oil_symbols
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

# Typed API helper functions
def get_recent_signals() -> List[SignalDTO]:
    """Get recent signals with proper typing"""
    result = call_api("/api/signals/recent?limit=10")
    if isinstance(result, list):
        # Filter to only dict entries and validate structure
        return [s for s in result if isinstance(s, dict)]
    return []

def get_risk_status() -> RiskStatusDTO:
    """Get risk status with proper typing"""
    result = call_api("/api/risk/status")
    if isinstance(result, dict):
        return result
    return {"kill_switch": False, "daily_loss_limit": 0.0, "current_loss": 0.0}

def get_stats() -> StatsDTO:
    """Get stats with proper typing"""
    result = call_api("/api/signals/stats")
    if isinstance(result, dict):
        return result
    return {"total_signals": 0, "active_signals": 0, "win_rate": 0.0}

# Load data
@st.cache_data(ttl=30)
def load_dashboard_data():
    """Load all dashboard data with proper typing"""
    return {
        "signals": get_recent_signals(),
        "risk_status": get_risk_status(),
        "stats": get_stats()
    }

# Clear cache if auto-refresh is enabled to ensure fresh data
if st.session_state.auto_refresh_enabled:
    # Only clear cache if we haven't refreshed recently to avoid excessive API calls
    if time.time() - st.session_state.last_refresh > 25:  # Clear cache 5 seconds before refresh
        st.cache_data.clear()

# Load dashboard data
data = load_dashboard_data()
signals: List[SignalDTO] = data.get("signals", [])
risk_status: RiskStatusDTO = data.get("risk_status", {"kill_switch": False, "daily_loss_limit": 0.0, "current_loss": 0.0})
stats: StatsDTO = data.get("stats", {"total_signals": 0, "active_signals": 0, "win_rate": 0.0})

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

# System Activity Timestamps
st.markdown('<div class="section-header">⏰ System Activity Monitor</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    # Last signal generation time
    if signals:
        latest_signal = max(signals, key=lambda s: s.get('issued_at', '2000-01-01T00:00:00Z'))
        signal_time = latest_signal.get('issued_at', 'Unknown')
        if signal_time and signal_time != 'Unknown':
            try:
                # Convert to Saudi time and format
                saudi_time = format_saudi_time(signal_time, "%H:%M:%S AST")
                signal_status = f"🟢 {saudi_time}"
            except:
                signal_status = "⚠️ Invalid"
        else:
            signal_status = "❌ None"
    else:
        signal_status = "❌ No Signals"
    st.metric("Last Signal Generated", signal_status)

with col2:
    # Market data freshness
    current_time = datetime.now()
    market_time = current_time.strftime('%H:%M:%S')
    if market_open:
        market_data_status = f"🟢 {market_time}"
    else:
        market_data_status = f"🔴 Market Closed"
    st.metric("Market Data", market_data_status)

with col3:
    # System uptime/activity
    system_activity = "🟢 Running" if not kill_switch else "🔴 Stopped"
    st.metric("System Status", system_activity)

with col4:
    # Signal generation frequency
    if signals:
        recent_signals = [s for s in signals if s.get('issued_at', '')]
        if len(recent_signals) >= 2:
            # Calculate time between last two signals
            times = sorted([s['issued_at'] for s in recent_signals if s.get('issued_at')], reverse=True)
            if len(times) >= 2:
                try:
                    latest = datetime.fromisoformat(times[0].replace('Z', '+00:00'))
                    previous = datetime.fromisoformat(times[1].replace('Z', '+00:00'))
                    diff = latest - previous
                    minutes = int(diff.total_seconds() / 60)
                    if minutes < 60:
                        freq_status = f"⚡ {minutes}m ago"
                    else:
                        hours = minutes // 60
                        freq_status = f"🕐 {hours}h ago"
                except:
                    freq_status = "⚠️ Error"
            else:
                freq_status = "📊 Active"
        else:
            freq_status = "⏳ Starting"
    else:
        freq_status = "❌ Inactive"
    st.metric("Signal Frequency", freq_status)

st.markdown("---")

# Separate signals by type
def separate_signals_by_type(signals: Sequence[Mapping[str, Any]]):
    """Separate signals into Forex Major, Crypto, and Metals & Oil categories"""
    # All 26 forex symbols that the backend processes (from scheduler.py)
    forex_majors = [
        # USD Major Pairs
        'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
        # EUR Cross Pairs
        'EURGBP', 'EURJPY', 'EURCHF', 'EURAUD', 'EURCAD',
        # GBP Cross Pairs
        'GBPJPY', 'GBPAUD', 'GBPCHF', 'GBPCAD',
        # JPY Cross Pairs
        'AUDJPY', 'CADJPY', 'CHFJPY', 'NZDJPY',
        # Other Major Cross Pairs
        'AUDCAD', 'AUDCHF', 'AUDNZD', 'CADCHF', 'NZDCAD', 'NZDCHF'
    ]
    # All 8 crypto symbols that the backend processes (from scheduler.py)
    crypto_pairs = ['BTCUSD', 'ETHUSD', 'ADAUSD', 'DOGEUSD', 'SOLUSD', 'BNBUSD', 'XRPUSD', 'MATICUSD']
    # All 8 metals & oil symbols that the backend processes (from scheduler.py)
    metals_oil = ['XAUUSD', 'XAGUSD', 'XPTUSD', 'XPDUSD', 'USOIL', 'UKOUSD', 'WTIUSD', 'XBRUSD']
    
    # Filter to only dict items and safely access symbol
    valid_signals = [s for s in signals if isinstance(s, dict)]
    
    forex_signals = [s for s in valid_signals if s.get('symbol', '').upper() in forex_majors]
    crypto_signals = [s for s in valid_signals if s.get('symbol', '').upper() in crypto_pairs]
    metals_oil_signals = [s for s in valid_signals if s.get('symbol', '').upper() in metals_oil]
    
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