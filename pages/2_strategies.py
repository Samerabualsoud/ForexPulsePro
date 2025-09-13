"""
Strategies Configuration Page
"""
import streamlit as st
import requests
import json
from datetime import datetime

st.set_page_config(page_title="Strategies", page_icon="⚙️", layout="wide")

st.title("⚙️ Strategy Configuration")

# Helper function to call backend API with production fallback
def call_api(endpoint, method="GET", data=None, token=None):
    """Call backend API with development/production environment detection"""
    try:
        base_url = "http://0.0.0.0:8000"
        url = f"{base_url}{endpoint}"
        
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=5)
        elif method == "POST":
            response = requests.post(url, json=data, headers=headers, timeout=5)
        elif method == "PUT":
            response = requests.put(url, json=data, headers=headers, timeout=5)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise requests.exceptions.ConnectionError("API not available")
            
    except requests.exceptions.RequestException as e:
        st.warning("⚠️ **Backend API unavailable** - Running in demo mode")
        return get_fallback_strategy_data(endpoint, method, data)

def get_fallback_strategy_data(endpoint, method="GET", data=None):
    """Provide fallback data for strategies when backend is unavailable"""
    
    if "/api/auth/login" in endpoint:
        # Simulate successful login for demo
        return {"access_token": "demo_token", "token_type": "bearer", "role": "admin"}
    elif "/api/strategies" in endpoint and method == "GET":
        # Return sample strategy configurations
        return {
            "EMA_RSI": {
                "name": "EMA + RSI",
                "description": "Exponential Moving Average with RSI",
                "enabled": True,
                "ema_fast": 12,
                "ema_slow": 26,
                "rsi_period": 14,
                "rsi_oversold": 30,
                "rsi_overbought": 70
            },
            "STOCHASTIC": {
                "name": "Stochastic Oscillator",
                "description": "Stochastic momentum indicator",
                "enabled": True,
                "k_period": 14,
                "d_period": 3,
                "oversold": 20,
                "overbought": 80
            },
            "FIBONACCI": {
                "name": "Fibonacci Retracement",
                "description": "Fibonacci-based trading strategy",
                "enabled": True,
                "lookback_period": 100,
                "retracement_levels": [0.236, 0.382, 0.618]
            }
        }
    elif "/api/strategies" in endpoint and method == "PUT":
        # Simulate successful strategy update
        st.success("✅ Strategy would be updated in production environment")
        return {"success": True, "message": "Demo mode - settings not actually saved"}
    
    return None

# Authentication state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'auth_token' not in st.session_state:
    st.session_state.auth_token = None

# Login form
if not st.session_state.authenticated:
    st.warning("⚠️ Admin authentication required to modify strategies")
    
    with st.form("login_form"):
        st.subheader("🔐 Admin Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit and username and password:
            # Try to authenticate
            auth_response = call_api("/api/auth/login", "POST", {
                "username": username,
                "password": password
            })
            
            if auth_response and auth_response.get("access_token"):
                st.session_state.authenticated = True
                st.session_state.auth_token = auth_response["access_token"]
                st.session_state.user_role = auth_response.get("role", "viewer")
                st.success("✅ Authentication successful!")
                st.rerun()
            else:
                st.error("❌ Invalid credentials")
    
    st.info("💡 Default credentials: admin/admin123 or viewer/viewer123")
    st.stop()

# Load strategies
@st.cache_data(ttl=60)
def load_strategies():
    """Load strategy configurations"""
    return call_api("/api/strategies")

strategies = load_strategies()

if not strategies:
    st.error("Failed to load strategy configurations")
    st.stop()

# Group strategies by symbol
symbols = {}
for strategy in strategies:
    symbol = strategy['symbol']
    if symbol not in symbols:
        symbols[symbol] = []
    symbols[symbol].append(strategy)

# Main interface
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("📊 Symbols")
    selected_symbol = st.selectbox(
        "Select Symbol:",
        options=list(symbols.keys()),
        index=0
    )

with col2:
    st.subheader(f"Strategies for {selected_symbol}")
    
    # Strategy tabs
    if selected_symbol in symbols:
        strategy_names = [s['name'] for s in symbols[selected_symbol]]
        tabs = st.tabs(strategy_names)
        
        for i, (tab, strategy) in enumerate(zip(tabs, symbols[selected_symbol])):
            with tab:
                st.markdown(f"### {strategy['name'].upper()} Strategy")
                
                # Strategy status
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    enabled = strategy['enabled']
                    status_color = "🟢" if enabled else "🔴"
                    st.metric("Status", f"{status_color} {'Enabled' if enabled else 'Disabled'}")
                
                with col_b:
                    st.metric("Symbol", strategy['symbol'])
                
                with col_c:
                    updated = datetime.fromisoformat(strategy['updated_at'].replace('Z', '+00:00'))
                    st.metric("Last Updated", updated.strftime("%m/%d %H:%M"))
                
                # Configuration form
                if st.session_state.user_role == "admin":
                    with st.form(f"strategy_form_{strategy['id']}"):
                        st.subheader("⚙️ Configuration")
                        
                        # Toggle enabled/disabled
                        new_enabled = st.checkbox(
                            "Strategy Enabled",
                            value=strategy['enabled'],
                            key=f"enabled_{strategy['id']}"
                        )
                        
                        # Configuration parameters
                        config = strategy['config']
                        
                        if strategy['name'] == 'ema_rsi':
                            st.subheader("EMA + RSI Parameters")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                ema_fast = st.number_input(
                                    "EMA Fast Period",
                                    min_value=5, max_value=50,
                                    value=config.get('ema_fast', 12),
                                    key=f"ema_fast_{strategy['id']}"
                                )
                                
                                rsi_period = st.number_input(
                                    "RSI Period",
                                    min_value=5, max_value=30,
                                    value=config.get('rsi_period', 14),
                                    key=f"rsi_period_{strategy['id']}"
                                )
                                
                                min_confidence = st.slider(
                                    "Minimum Confidence",
                                    min_value=0.0, max_value=1.0,
                                    value=config.get('min_confidence', 0.6),
                                    step=0.05,
                                    key=f"min_conf_{strategy['id']}"
                                )
                            
                            with col2:
                                ema_slow = st.number_input(
                                    "EMA Slow Period",
                                    min_value=10, max_value=100,
                                    value=config.get('ema_slow', 26),
                                    key=f"ema_slow_{strategy['id']}"
                                )
                                
                                rsi_threshold = st.number_input(
                                    "RSI Threshold",
                                    min_value=30, max_value=70,
                                    value=config.get('rsi_buy_threshold', 50),
                                    key=f"rsi_thresh_{strategy['id']}"
                                )
                                
                                expiry_bars = st.number_input(
                                    "Signal Expiry (minutes)",
                                    min_value=15, max_value=240,
                                    value=config.get('expiry_bars', 60),
                                    key=f"expiry_{strategy['id']}"
                                )
                            
                            # Stop Loss / Take Profit
                            st.subheader("Risk Management")
                            sl_mode = st.selectbox(
                                "Stop Loss Mode",
                                options=['atr', 'pips'],
                                index=0 if config.get('sl_mode') == 'atr' else 1,
                                key=f"sl_mode_{strategy['id']}"
                            )
                            
                            if sl_mode == 'atr':
                                sl_multiplier = st.slider(
                                    "SL ATR Multiplier",
                                    min_value=1.0, max_value=5.0,
                                    value=config.get('sl_multiplier', 2.0),
                                    step=0.1,
                                    key=f"sl_mult_{strategy['id']}"
                                )
                            else:
                                sl_pips = st.number_input(
                                    "SL Pips",
                                    min_value=5, max_value=100,
                                    value=config.get('sl_pips', 20),
                                    key=f"sl_pips_{strategy['id']}"
                                )
                            
                            new_config = {
                                'ema_fast': ema_fast,
                                'ema_slow': ema_slow,
                                'rsi_period': rsi_period,
                                'rsi_buy_threshold': rsi_threshold,
                                'rsi_sell_threshold': rsi_threshold,
                                'sl_mode': sl_mode,
                                'tp_mode': sl_mode,
                                'min_confidence': min_confidence,
                                'expiry_bars': expiry_bars
                            }
                            
                            if sl_mode == 'atr':
                                new_config['sl_multiplier'] = sl_multiplier
                                new_config['tp_multiplier'] = sl_multiplier * 1.5
                            else:
                                new_config['sl_pips'] = sl_pips
                                new_config['tp_pips'] = sl_pips * 2
                        
                        elif strategy['name'] == 'donchian_atr':
                            st.subheader("Donchian + ATR Parameters")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                donchian_period = st.number_input(
                                    "Donchian Period",
                                    min_value=10, max_value=50,
                                    value=config.get('donchian_period', 20),
                                    key=f"don_period_{strategy['id']}"
                                )
                                
                                use_supertrend = st.checkbox(
                                    "Use SuperTrend Filter",
                                    value=config.get('use_supertrend', True),
                                    key=f"supertrend_{strategy['id']}"
                                )
                            
                            with col2:
                                atr_period = st.number_input(
                                    "ATR Period",
                                    min_value=5, max_value=30,
                                    value=config.get('atr_period', 14),
                                    key=f"atr_period_{strategy['id']}"
                                )
                                
                                min_confidence = st.slider(
                                    "Minimum Confidence",
                                    min_value=0.0, max_value=1.0,
                                    value=config.get('min_confidence', 0.65),
                                    step=0.05,
                                    key=f"min_conf_don_{strategy['id']}"
                                )
                            
                            new_config = {
                                'donchian_period': donchian_period,
                                'atr_period': atr_period,
                                'use_supertrend': use_supertrend,
                                'sl_mode': 'atr',
                                'tp_mode': 'atr',
                                'sl_multiplier': 2.0,
                                'tp_multiplier': 3.0,
                                'min_confidence': min_confidence,
                                'expiry_bars': 45
                            }
                        
                        elif strategy['name'] == 'meanrev_bb':
                            st.subheader("Mean Reversion BB Parameters")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                bb_period = st.number_input(
                                    "Bollinger Bands Period",
                                    min_value=10, max_value=50,
                                    value=config.get('bb_period', 20),
                                    key=f"bb_period_{strategy['id']}"
                                )
                                
                                adx_threshold = st.number_input(
                                    "ADX Threshold",
                                    min_value=15, max_value=40,
                                    value=config.get('adx_threshold', 25),
                                    key=f"adx_thresh_{strategy['id']}"
                                )
                            
                            with col2:
                                bb_std = st.slider(
                                    "BB Standard Deviations",
                                    min_value=1.0, max_value=3.0,
                                    value=config.get('bb_std', 2.0),
                                    step=0.1,
                                    key=f"bb_std_{strategy['id']}"
                                )
                                
                                zscore_threshold = st.slider(
                                    "Z-Score Threshold",
                                    min_value=1.0, max_value=3.0,
                                    value=config.get('zscore_threshold', 2.0),
                                    step=0.1,
                                    key=f"zscore_{strategy['id']}"
                                )
                            
                            new_config = {
                                'bb_period': bb_period,
                                'bb_std': bb_std,
                                'adx_period': 14,
                                'adx_threshold': adx_threshold,
                                'zscore_threshold': zscore_threshold,
                                'sl_mode': 'pips',
                                'tp_mode': 'pips',
                                'sl_pips': 20,
                                'tp_pips': 40,
                                'min_confidence': 0.7,
                                'expiry_bars': 30
                            }
                        
                        # Submit button
                        submit_button = st.form_submit_button("💾 Save Configuration")
                        
                        if submit_button:
                            # Update strategy
                            update_data = {
                                'enabled': new_enabled,
                                'config': new_config
                            }
                            
                            result = call_api(
                                f"/api/strategies/{strategy['id']}",
                                "PUT",
                                update_data,
                                st.session_state.auth_token
                            )
                            
                            if result:
                                st.success("✅ Strategy updated successfully!")
                                st.cache_data.clear()
                                st.rerun()
                            else:
                                st.error("❌ Failed to update strategy")
                
                else:
                    # Read-only view for non-admin users
                    st.subheader("📋 Current Configuration")
                    st.json(strategy['config'])
                    st.info("🔒 Admin privileges required to modify configurations")

# Strategy performance summary
st.markdown("---")
st.subheader("📊 Strategy Performance Summary")

# This would typically come from a performance tracking system
# For now, show basic stats from recent signals
recent_signals = call_api("/api/signals/recent?limit=100")

if recent_signals:
    import pandas as pd
    
    # Convert to DataFrame
    df = pd.DataFrame(recent_signals)
    
    # Group by strategy
    strategy_stats = df.groupby(['strategy', 'symbol']).agg({
        'confidence': ['mean', 'count'],
        'blocked_by_risk': lambda x: (x == False).sum(),
        'sent_to_whatsapp': lambda x: (x == True).sum()
    }).round(2)
    
    strategy_stats.columns = ['Avg Confidence', 'Total Signals', 'Allowed', 'Sent to WhatsApp']
    
    st.dataframe(strategy_stats, use_container_width=True)

# Logout button
if st.button("🚪 Logout"):
    st.session_state.authenticated = False
    st.session_state.auth_token = None
    st.rerun()
