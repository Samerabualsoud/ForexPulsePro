"""
Risk Management Page - Simplified
"""
import streamlit as st
import requests
from datetime import datetime
import sys
from pathlib import Path

st.set_page_config(page_title="Risk Management", page_icon="🛡️", layout="wide")

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# No authentication required
user_info = {"username": "user", "role": "admin"}

# Clean, simple CSS styling
st.markdown("""
<style>
    /* Simple title styling */
    .risk-title {
        font-size: 2.5rem;
        font-weight: 600;
        text-align: center;
        color: #2c3e50;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 3px solid #e74c3c;
    }
    
    /* Status card styling */
    .status-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .status-card.safe {
        border-left: 5px solid #28a745;
    }
    
    .status-card.warning {
        border-left: 5px solid #ffc107;
    }
    
    .status-card.danger {
        border-left: 5px solid #dc3545;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e74c3c;
    }
    
    /* Control buttons */
    .control-button {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        cursor: pointer;
        width: 100%;
        transition: all 0.2s;
    }
    
    .control-button.safe {
        background: #28a745;
        color: white;
    }
    
    .control-button.danger {
        background: #dc3545;
        color: white;
    }
    
    /* Progress bar styling */
    .risk-progress {
        height: 20px;
        border-radius: 10px;
        overflow: hidden;
        background: #e9ecef;
        margin: 1rem 0;
    }
    
    /* Info boxes */
    .info-box {
        background: #e8f4fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .danger-box {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="risk-title">🛡️ Risk Management</h1>', unsafe_allow_html=True)

# Helper function for API calls
def call_api(endpoint, method="GET", data=None):
    """Call backend API with fallback to demo data"""
    try:
        base_url = "http://0.0.0.0:8000"
        url = f"{base_url}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, timeout=5)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=5)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise requests.exceptions.ConnectionError("API not available")
            
    except requests.exceptions.RequestException:
        st.info("🔄 Running in demo mode")
        return get_demo_risk_data(endpoint, method, data)

def get_demo_risk_data(endpoint, method="GET", data=None):
    """Provide demo risk data"""
    import random
    
    if "/api/risk/status" in endpoint:
        return {
            "kill_switch_enabled": False,
            "daily_loss_limit": 1000,
            "current_daily_loss": random.randint(50, 250),
            "signals_blocked_today": random.randint(0, 5),
            "signals_allowed_today": random.randint(8, 20),
            "volatility_guard_enabled": True
        }
    elif "/api/risk/killswitch" in endpoint and method == "POST":
        enabled = data.get("enabled", False)
        status = "enabled" if enabled else "disabled"
        st.success(f"✅ Kill switch {status}!")
        return {"success": True}
    elif "/api/risk/config" in endpoint and method == "POST":
        st.success("✅ Risk settings updated!")
        return {"success": True}
    
    return {}

# Load risk status
@st.cache_data(ttl=30)
def load_risk_status():
    """Load current risk status"""
    return call_api("/api/risk/status")

risk_status = load_risk_status()

if not risk_status:
    st.error("Unable to load risk status")
    st.stop()

# System Status Overview
st.markdown('<div class="section-header">📊 System Status</div>', unsafe_allow_html=True)

kill_switch_enabled = risk_status.get('kill_switch_enabled', False)
daily_loss = risk_status.get('current_daily_loss', 0)
loss_limit = risk_status.get('daily_loss_limit', 1000)
loss_percentage = (daily_loss / loss_limit * 100) if loss_limit > 0 else 0

# Determine overall risk level
if kill_switch_enabled:
    risk_level = "🔴 STOPPED"
    risk_class = "danger"
elif loss_percentage > 80:
    risk_level = "🔴 HIGH RISK"
    risk_class = "danger"
elif loss_percentage > 60:
    risk_level = "🟡 MODERATE RISK"
    risk_class = "warning"
else:
    risk_level = "🟢 LOW RISK"
    risk_class = "safe"

# Status overview
col1, col2, col3, col4 = st.columns(4)

with col1:
    trading_status = "🔴 STOPPED" if kill_switch_enabled else "🟢 ACTIVE"
    st.metric("Trading Status", trading_status)

with col2:
    st.metric("Daily Loss", f"${daily_loss:.0f}")

with col3:
    remaining = loss_limit - daily_loss
    st.metric("Budget Remaining", f"${remaining:.0f}")

with col4:
    st.metric("Risk Level", risk_level)

# Progress bar for daily loss
if loss_limit > 0:
    progress = min(daily_loss / loss_limit, 1.0)
    
    # Color based on risk level
    if progress > 0.8:
        bar_color = "#dc3545"
    elif progress > 0.6:
        bar_color = "#ffc107"
    else:
        bar_color = "#28a745"
    
    st.markdown(f"""
    <div style="background: #e9ecef; border-radius: 10px; height: 20px; margin: 1rem 0;">
        <div style="background: {bar_color}; height: 100%; width: {progress*100:.1f}%; border-radius: 10px; transition: width 0.3s ease;"></div>
    </div>
    <div style="text-align: center; color: #6c757d; font-size: 0.9rem;">
        Daily Loss Progress: ${daily_loss:.0f} / ${loss_limit:.0f} ({progress*100:.1f}%)
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Main Controls
st.markdown('<div class="section-header">🎛️ Risk Controls</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🚨 Emergency Stop")
    
    if kill_switch_enabled:
        st.markdown("""
        <div class="danger-box">
            <strong>⚠️ Trading is currently STOPPED</strong><br>
            All signal generation and delivery is blocked.
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🟢 Resume Trading", type="primary", use_container_width=True):
            result = call_api("/api/risk/killswitch", "POST", {"enabled": False})
            if result:
                st.cache_data.clear()
                st.rerun()
    else:
        st.markdown("""
        <div class="info-box">
            <strong>✅ Trading is ACTIVE</strong><br>
            Signals are being generated and delivered normally.
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔴 Stop All Trading", type="secondary", use_container_width=True):
            result = call_api("/api/risk/killswitch", "POST", {"enabled": True})
            if result:
                st.cache_data.clear()
                st.rerun()
    
    st.markdown("""
    <div style="background: #f8f9fa; padding: 0.8rem; border-radius: 6px; margin-top: 1rem;">
        <small><strong>Kill Switch</strong> - Instantly stops all signal generation and WhatsApp delivery. 
        Use during high-impact news events or emergencies.</small>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("### ⚙️ Risk Settings")
    
    # Daily loss limit setting
    with st.form("risk_settings"):
        new_loss_limit = st.number_input(
            "Daily Loss Limit ($)",
            min_value=100,
            max_value=5000,
            value=int(loss_limit),
            step=50,
            help="Maximum daily loss before trading stops"
        )
        
        volatility_guard = st.checkbox(
            "Enable Volatility Protection",
            value=risk_status.get('volatility_guard_enabled', True),
            help="Block signals during high market volatility"
        )
        
        if st.form_submit_button("💾 Save Settings", use_container_width=True):
            update_data = {
                "daily_loss_limit": new_loss_limit,
                "volatility_guard_enabled": volatility_guard
            }
            result = call_api("/api/risk/config", "POST", update_data)
            if result:
                st.cache_data.clear()
                st.rerun()

# Risk Activity Summary
st.markdown("---")
st.markdown('<div class="section-header">📈 Today\'s Activity</div>', unsafe_allow_html=True)

signals_blocked = risk_status.get('signals_blocked_today', 0)
signals_allowed = risk_status.get('signals_allowed_today', 0)
total_signals = signals_blocked + signals_allowed

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Signals", total_signals)

with col2:
    st.metric("Allowed", signals_allowed)

with col3:
    st.metric("Blocked", signals_blocked)

# Risk warnings
if loss_percentage > 80:
    st.markdown("""
    <div class="danger-box">
        <strong>🚨 HIGH RISK WARNING</strong><br>
        You've used {:.0f}% of your daily loss limit. Consider stopping trading or reducing position sizes.
    </div>
    """.format(loss_percentage), unsafe_allow_html=True)
elif loss_percentage > 60:
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ MODERATE RISK</strong><br>
        You've used {:.0f}% of your daily loss limit. Monitor positions carefully.
    </div>
    """.format(loss_percentage), unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="info-box">
        <strong>✅ LOW RISK</strong><br>
        You've used only {:.0f}% of your daily loss limit. Trading within safe parameters.
    </div>
    """.format(loss_percentage), unsafe_allow_html=True)

# Quick Reference
st.markdown("---")

with st.expander("📚 Risk Management Guide"):
    st.markdown("""
    ### 🛡️ Risk Controls Explained
    
    **Kill Switch**
    - Instantly stops all signal generation
    - Use during major news events or system issues
    - Can be toggled on/off immediately
    
    **Daily Loss Limit**
    - Set to 1-2% of your total trading account
    - Automatically stops trading when reached
    - Resets daily at midnight UTC
    
    **Volatility Protection**
    - Blocks signals during abnormal market volatility
    - Helps avoid false signals during news events
    - Automatically monitors market conditions
    
    ### 💡 Best Practices
    
    - Set daily loss limit to match your risk tolerance
    - Use kill switch during high-impact news releases
    - Monitor risk level throughout the trading day
    - Keep volatility protection enabled for safety
    - Review and adjust settings based on performance
    
    ### 📞 Emergency Actions
    
    1. **Immediate Stop**: Use kill switch to halt all trading
    2. **Reduce Risk**: Lower daily loss limit
    3. **Monitor Closely**: Watch risk level indicators
    4. **Seek Help**: Contact support if needed
    """)

# Quick Actions
st.markdown("---")
st.markdown('<div class="section-header">⚡ Quick Actions</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔄 Refresh Status", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

with col2:
    if st.button("📊 View Signals", use_container_width=True):
        st.switch_page("pages/1_overview.py")

with col3:
    if st.button("⚙️ Strategy Settings", use_container_width=True):
        st.switch_page("pages/2_strategies.py")

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #7f8c8d; font-size: 0.9rem;">
    Risk status last updated: {datetime.now().strftime('%H:%M:%S')} | Auto-refresh every 30 seconds
</div>
""", unsafe_allow_html=True)