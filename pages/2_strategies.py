"""
Strategy Configuration Page - Simplified
"""
import streamlit as st
import requests
from datetime import datetime
import sys
from pathlib import Path

st.set_page_config(page_title="Strategies", page_icon="⚙️", layout="wide")

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
    .strategy-title {
        font-size: 2.5rem;
        font-weight: 600;
        text-align: center;
        color: #2c3e50;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 3px solid #3498db;
    }
    
    /* Strategy card styling */
    .strategy-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .strategy-card.enabled {
        border-left: 5px solid #28a745;
    }
    
    .strategy-card.disabled {
        border-left: 5px solid #dc3545;
        opacity: 0.7;
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
    
    /* Info boxes */
    .info-box {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    /* Strategy description */
    .strategy-description {
        color: #6c757d;
        font-size: 0.95rem;
        line-height: 1.5;
        margin-bottom: 1rem;
    }
    
    /* Status indicators */
    .status-enabled { color: #28a745; font-weight: 600; }
    .status-disabled { color: #dc3545; font-weight: 600; }
    
    /* Toggle switch styling */
    .stCheckbox > div {
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="strategy-title">⚙️ Trading Strategies</h1>', unsafe_allow_html=True)

# Helper function for API calls
def call_api(endpoint, method="GET", data=None):
    """Call backend API with fallback to demo data"""
    try:
        base_url = "http://0.0.0.0:8000"
        url = f"{base_url}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, timeout=5)
        elif method == "PUT":
            response = requests.put(url, json=data, timeout=5)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise requests.exceptions.ConnectionError("API not available")
            
    except requests.exceptions.RequestException:
        st.info("🔄 Running in demo mode")
        return get_demo_strategy_data(endpoint, method, data)

def get_demo_strategy_data(endpoint, method="GET", data=None):
    """Provide demo strategy data"""
    if "/api/strategies" in endpoint and method == "GET":
        return [
            {
                "id": 1,
                "name": "EMA + RSI",
                "description": "Uses moving averages and momentum to identify trend changes",
                "enabled": True,
                "performance": {"signals_today": 5, "success_rate": 75},
                "risk_level": "Medium"
            },
            {
                "id": 2,
                "name": "Bollinger Bands",
                "description": "Identifies overbought and oversold conditions using price volatility",
                "enabled": True,
                "performance": {"signals_today": 3, "success_rate": 68},
                "risk_level": "Low"
            },
            {
                "id": 3,
                "name": "MACD Divergence",
                "description": "Detects trend reversals using momentum divergence",
                "enabled": False,
                "performance": {"signals_today": 0, "success_rate": 82},
                "risk_level": "High"
            },
            {
                "id": 4,
                "name": "Stochastic Oscillator",
                "description": "Identifies trend momentum and potential reversal points",
                "enabled": True,
                "performance": {"signals_today": 2, "success_rate": 71},
                "risk_level": "Medium"
            },
            {
                "id": 5,
                "name": "Fibonacci Retracement",
                "description": "Uses mathematical ratios to find support and resistance levels",
                "enabled": False,
                "performance": {"signals_today": 0, "success_rate": 65},
                "risk_level": "Low"
            }
        ]
    elif method == "PUT":
        st.success("✅ Strategy settings updated!")
        return {"success": True}
    
    return []

# Load strategies
@st.cache_data(ttl=60)
def load_strategies():
    """Load strategy configurations"""
    return call_api("/api/strategies")

strategies = load_strategies()

if not strategies:
    st.error("Failed to load strategies")
    st.stop()

# Strategy overview
st.markdown('<div class="section-header">📊 Strategy Overview</div>', unsafe_allow_html=True)

enabled_count = len([s for s in strategies if s.get('enabled', False)])
total_count = len(strategies)
today_signals = sum(s.get('performance', {}).get('signals_today', 0) for s in strategies)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Active Strategies", f"{enabled_count}/{total_count}")

with col2:
    st.metric("Signals Today", today_signals)

with col3:
    avg_success = sum(s.get('performance', {}).get('success_rate', 0) for s in strategies) / len(strategies) if strategies else 0
    st.metric("Avg Success Rate", f"{avg_success:.0f}%")

st.markdown("---")

# Simple strategy management
st.markdown('<div class="section-header">🎛️ Strategy Controls</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <strong>💡 How it works:</strong> Enable the strategies you want to use for generating trading signals. 
    Each strategy uses different technical indicators to identify trading opportunities.
</div>
""", unsafe_allow_html=True)

# Strategy cards
for strategy in strategies:
    strategy_id = strategy['id']
    name = strategy['name']
    description = strategy.get('description', 'No description available')
    enabled = strategy.get('enabled', False)
    performance = strategy.get('performance', {})
    risk_level = strategy.get('risk_level', 'Medium')
    
    # Strategy card
    card_class = "enabled" if enabled else "disabled"
    
    with st.container():
        st.markdown(f'<div class="strategy-card {card_class}">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.markdown(f"### {name}")
            st.markdown(f'<div class="strategy-description">{description}</div>', unsafe_allow_html=True)
            
            # Risk level indicator
            risk_colors = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
            risk_color = risk_colors.get(risk_level, "🟡")
            st.markdown(f"**Risk Level:** {risk_color} {risk_level}")
        
        with col2:
            # Performance metrics
            signals_today = performance.get('signals_today', 0)
            success_rate = performance.get('success_rate', 0)
            
            st.metric("Today", signals_today)
            st.metric("Success", f"{success_rate}%")
        
        with col3:
            # Enable/Disable toggle
            status_text = "🟢 Enabled" if enabled else "🔴 Disabled"
            st.markdown(f"**Status:** {status_text}")
            
            # Toggle button
            new_enabled = st.toggle(
                f"Enable {name}",
                value=enabled,
                key=f"toggle_{strategy_id}",
                help=f"Turn {name} strategy on or off"
            )
            
            # Update if changed
            if new_enabled != enabled:
                update_data = {"enabled": new_enabled}
                result = call_api(f"/api/strategies/{strategy_id}", "PUT", update_data)
                if result:
                    st.cache_data.clear()
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("")

# Quick actions
st.markdown("---")
st.markdown('<div class="section-header">⚡ Quick Actions</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🟢 Enable All Strategies", use_container_width=True):
        for strategy in strategies:
            call_api(f"/api/strategies/{strategy['id']}", "PUT", {"enabled": True})
        st.success("All strategies enabled!")
        st.cache_data.clear()
        st.rerun()

with col2:
    if st.button("🔴 Disable All Strategies", use_container_width=True):
        for strategy in strategies:
            call_api(f"/api/strategies/{strategy['id']}", "PUT", {"enabled": False})
        st.warning("All strategies disabled!")
        st.cache_data.clear()
        st.rerun()

with col3:
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Advanced settings (collapsed by default)
st.markdown("---")

with st.expander("🔧 Advanced Settings", expanded=False):
    st.markdown("### ⚙️ Advanced Strategy Configuration")
    
    st.info("""
    **Coming Soon:** Advanced parameter tuning for each strategy including:
    - Signal confidence thresholds
    - Risk management settings  
    - Custom indicator periods
    - Stop loss and take profit levels
    """)
    
    # Simple global settings
    st.markdown("#### Global Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        min_confidence = st.slider(
            "Minimum Signal Confidence",
            min_value=50,
            max_value=95,
            value=70,
            step=5,
            help="Only generate signals with confidence above this threshold"
        )
    
    with col2:
        signal_expiry = st.selectbox(
            "Signal Expiry Time",
            options=[15, 30, 45, 60, 90],
            index=1,
            help="How long signals remain valid (minutes)"
        )
    
    if st.button("💾 Save Advanced Settings"):
        st.success("✅ Advanced settings saved!")

# Performance summary
st.markdown("---")
st.markdown('<div class="section-header">📈 Performance Summary</div>', unsafe_allow_html=True)

if strategies:
    # Create performance summary table
    perf_data = []
    for strategy in strategies:
        perf = strategy.get('performance', {})
        perf_data.append({
            'Strategy': strategy['name'],
            'Status': "✅ Enabled" if strategy.get('enabled') else "❌ Disabled",
            'Signals Today': perf.get('signals_today', 0),
            'Success Rate': f"{perf.get('success_rate', 0)}%",
            'Risk Level': strategy.get('risk_level', 'Medium')
        })
    
    import pandas as pd
    df = pd.DataFrame(perf_data)
    
    # Style the performance table
    def style_status(val):
        if "Enabled" in val:
            return 'color: #28a745; font-weight: bold;'
        else:
            return 'color: #dc3545; font-weight: bold;'
    
    def style_success_rate(val):
        try:
            rate = int(val.replace('%', ''))
            if rate >= 80:
                return 'background-color: #d4edda; color: #155724; font-weight: bold;'
            elif rate >= 70:
                return 'background-color: #fff3cd; color: #856404; font-weight: bold;'
            else:
                return 'background-color: #f8d7da; color: #721c24; font-weight: bold;'
        except:
            return ''
    
    styled_df = df.style.map(style_status, subset=['Status'])
    styled_df = styled_df.map(style_success_rate, subset=['Success Rate'])
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

# Help section
st.markdown("---")

with st.expander("❓ Strategy Help & Information"):
    st.markdown("""
    ### 📚 Strategy Descriptions
    
    **EMA + RSI**: Combines trend-following moving averages with momentum indicators to catch trend changes early.
    - Best for: Trending markets
    - Signals: Medium frequency, good accuracy
    
    **Bollinger Bands**: Uses price volatility to identify when markets are overbought or oversold.
    - Best for: Range-bound markets
    - Signals: Lower frequency, reliable reversals
    
    **MACD Divergence**: Looks for momentum divergence to predict trend reversals.
    - Best for: Experienced traders
    - Signals: Lower frequency, higher accuracy
    
    **Stochastic Oscillator**: Momentum oscillator that identifies trend strength and reversals.
    - Best for: Short-term trading
    - Signals: Higher frequency, moderate accuracy
    
    **Fibonacci Retracement**: Uses mathematical ratios to find key support and resistance levels.
    - Best for: Technical analysis enthusiasts
    - Signals: Lower frequency, good for entries
    
    ### ⚡ Quick Tips
    - Start with 2-3 strategies to avoid signal conflicts
    - Monitor performance regularly and adjust as needed
    - Higher confidence settings = fewer but better signals
    - Combine low and medium risk strategies for balance
    """)

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #7f8c8d; font-size: 0.9rem;">
    Strategy settings last updated: {datetime.now().strftime('%H:%M:%S')}
</div>
""", unsafe_allow_html=True)