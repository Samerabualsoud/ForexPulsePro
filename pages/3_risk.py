"""
Risk Management Page
"""
import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Risk Management", page_icon="🛡️", layout="wide")

st.title("🛡️ Risk Management")

# Helper function to call backend API
def call_api(endpoint, method="GET", data=None, token=None):
    """Call backend API"""
    try:
        base_url = "http://localhost:8000"
        url = f"{base_url}{endpoint}"
        
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, headers=headers, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
        return None

# Authentication state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'auth_token' not in st.session_state:
    st.session_state.auth_token = None

# Load risk status
@st.cache_data(ttl=30)  # Cache for 30 seconds
def load_risk_status():
    """Load current risk status"""
    return call_api("/api/risk/status")

risk_status = load_risk_status()

if not risk_status:
    st.error("Failed to load risk management status")
    st.stop()

# Current Status Overview
st.subheader("🔍 Current Risk Status")

col1, col2, col3, col4 = st.columns(4)

with col1:
    kill_switch = risk_status.get('kill_switch_enabled', False)
    status_color = "🔴" if kill_switch else "🟢"
    st.metric(
        "Kill Switch",
        f"{status_color} {'ACTIVE' if kill_switch else 'INACTIVE'}",
        delta="BLOCKING SIGNALS" if kill_switch else "ALLOWING SIGNALS"
    )

with col2:
    volatility_guard = risk_status.get('volatility_guard_enabled', False)
    status_color = "🟢" if volatility_guard else "🔴"
    st.metric(
        "Volatility Guard",
        f"{status_color} {'ON' if volatility_guard else 'OFF'}",
        delta="MONITORING" if volatility_guard else "DISABLED"
    )

with col3:
    daily_loss = risk_status.get('current_daily_loss', 0)
    loss_limit = risk_status.get('daily_loss_limit', 1000)
    st.metric(
        "Daily Loss",
        f"${daily_loss:.0f}",
        delta=f"${loss_limit - daily_loss:.0f} remaining"
    )

with col4:
    loss_percentage = (daily_loss / loss_limit * 100) if loss_limit > 0 else 0
    color = "🔴" if loss_percentage > 80 else "🟡" if loss_percentage > 60 else "🟢"
    st.metric(
        "Loss Utilization",
        f"{color} {loss_percentage:.1f}%",
        delta=f"of ${loss_limit:.0f} limit"
    )

# Progress bar for daily loss
if loss_limit > 0:
    progress = min(daily_loss / loss_limit, 1.0)
    st.progress(progress)
    st.caption(f"Daily Loss Progress: ${daily_loss:.0f} / ${loss_limit:.0f}")

st.markdown("---")

# Control Panel (Admin Only)
if not st.session_state.authenticated:
    st.warning("⚠️ Admin authentication required to modify risk settings")
    
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
else:
    # Admin Control Panel
    st.subheader("🎛️ Risk Controls")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚨 Emergency Controls")
        
        # Kill Switch
        current_kill_switch = risk_status.get('kill_switch_enabled', False)
        
        if current_kill_switch:
            if st.button("🟢 DISABLE Kill Switch", type="primary", use_container_width=True):
                result = call_api(
                    "/api/risk/killswitch",
                    "POST",
                    {"enabled": False},
                    st.session_state.auth_token
                )
                if result:
                    st.success("✅ Kill switch disabled - signals will resume")
                    st.cache_data.clear()
                    st.rerun()
        else:
            if st.button("🔴 ENABLE Kill Switch", type="secondary", use_container_width=True):
                result = call_api(
                    "/api/risk/killswitch",
                    "POST",
                    {"enabled": True},
                    st.session_state.auth_token
                )
                if result:
                    st.success("✅ Kill switch enabled - all signals blocked")
                    st.cache_data.clear()
                    st.rerun()
        
        st.info("ℹ️ Kill switch immediately stops all signal generation and WhatsApp delivery")
        
        # Manual signal resend
        st.subheader("📱 Manual Actions")
        
        if st.button("🧪 Test WhatsApp Connection", use_container_width=True):
            result = call_api(
                "/api/whatsapp/test",
                "POST",
                {},
                st.session_state.auth_token
            )
            if result:
                st.success("✅ WhatsApp test message sent successfully")
            else:
                st.error("❌ WhatsApp test failed")
    
    with col2:
        st.subheader("⚙️ Risk Parameters")
        
        with st.form("risk_config_form"):
            # Daily Loss Limit
            new_daily_limit = st.number_input(
                "Daily Loss Limit ($)",
                min_value=100,
                max_value=10000,
                value=int(risk_status.get('daily_loss_limit', 1000)),
                step=100
            )
            
            # Volatility Guard
            volatility_enabled = st.checkbox(
                "Enable Volatility Guard",
                value=risk_status.get('volatility_guard_enabled', True),
                help="Block signals when ATR exceeds threshold"
            )
            
            # Max Daily Signals
            max_daily_signals = st.number_input(
                "Maximum Daily Signals",
                min_value=10,
                max_value=500,
                value=100,
                step=10
            )
            
            if st.form_submit_button("💾 Save Risk Settings"):
                st.info("Risk parameter updates would be implemented via API")
                # This would call an API endpoint to update risk configuration
                # For now, showing as info since the endpoint isn't fully implemented

# Recent Risk Events
st.markdown("---")
st.subheader("📊 Risk Activity")

# Load recent signals with risk information
@st.cache_data(ttl=60)
def load_risk_events():
    """Load recent signals with risk blocking information"""
    return call_api("/api/signals/recent?limit=50")

risk_events = load_risk_events()

if risk_events:
    # Filter for risk-related events
    risk_blocked = [s for s in risk_events if s.get('blocked_by_risk')]
    allowed_signals = [s for s in risk_events if not s.get('blocked_by_risk')]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Signals Blocked Today", len(risk_blocked))
        
        if risk_blocked:
            st.subheader("🚫 Recently Blocked Signals")
            for event in risk_blocked[:5]:
                with st.container():
                    time_str = datetime.fromisoformat(event['issued_at'].replace('Z', '+00:00')).strftime("%H:%M")
                    st.write(f"🚫 **{event['symbol']}** {event['action']} blocked at {time_str}")
                    if event.get('risk_reason'):
                        st.caption(f"Reason: {event['risk_reason']}")
                    st.markdown("---")
    
    with col2:
        st.metric("Signals Allowed Today", len(allowed_signals))
        
        if allowed_signals:
            st.subheader("✅ Recently Allowed Signals")
            for event in allowed_signals[:5]:
                with st.container():
                    time_str = datetime.fromisoformat(event['issued_at'].replace('Z', '+00:00')).strftime("%H:%M")
                    whatsapp_status = "📱" if event.get('sent_to_whatsapp') else "⏳"
                    st.write(f"✅ **{event['symbol']}** {event['action']} @ {event['price']:.5f} {whatsapp_status}")
                    st.caption(f"Time: {time_str} • Confidence: {event['confidence']:.2f}")
                    st.markdown("---")

# Risk Statistics Chart
if risk_events:
    st.subheader("📈 Risk Statistics (Last 24 Hours)")
    
    # Create hourly risk statistics
    df = pd.DataFrame(risk_events)
    df['hour'] = pd.to_datetime(df['issued_at']).dt.hour
    
    hourly_stats = df.groupby('hour').agg({
        'blocked_by_risk': ['sum', 'count']
    }).reset_index()
    
    hourly_stats.columns = ['Hour', 'Blocked', 'Total']
    hourly_stats['Allowed'] = hourly_stats['Total'] - hourly_stats['Blocked']
    hourly_stats['Block_Rate'] = (hourly_stats['Blocked'] / hourly_stats['Total'] * 100).fillna(0)
    
    # Plot
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=hourly_stats['Hour'],
        y=hourly_stats['Allowed'],
        name='Allowed',
        marker_color='green',
        opacity=0.7
    ))
    
    fig.add_trace(go.Bar(
        x=hourly_stats['Hour'],
        y=hourly_stats['Blocked'],
        name='Blocked',
        marker_color='red',
        opacity=0.7
    ))
    
    fig.update_layout(
        title="Hourly Signal Risk Status",
        xaxis_title="Hour of Day",
        yaxis_title="Number of Signals",
        barmode='stack',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Risk Guidelines
st.markdown("---")
st.subheader("📚 Risk Management Guidelines")

with st.expander("🛡️ Risk Control Mechanisms"):
    st.markdown("""
    **Kill Switch**
    - Immediately stops all signal generation and WhatsApp delivery
    - Use during high-impact news events or system maintenance
    - Can be toggled instantly by admin users
    
    **Daily Loss Limit**
    - Estimated maximum daily loss tolerance
    - Blocks new signals when limit is approached
    - Resets daily at midnight UTC
    
    **Volatility Guard**
    - Monitors Average True Range (ATR) as percentage of price
    - Blocks signals during abnormally high volatility periods
    - Helps avoid false signals during news events
    
    **Signal Quality Filters**
    - Minimum confidence thresholds
    - Risk/reward ratio validation
    - Maximum daily signal limits
    """)

with st.expander("⚙️ Configuration Best Practices"):
    st.markdown("""
    **Daily Loss Limit**
    - Set to 1-2% of total account balance
    - Consider your risk tolerance and trading strategy
    - Monitor and adjust based on performance
    
    **Volatility Threshold**
    - Default 2% ATR is suitable for major pairs
    - Lower threshold (1.5%) for more conservative approach
    - Higher threshold (2.5%) for more aggressive trading
    
    **Signal Expiry**
    - Shorter expiry (15-30 min) for scalping strategies
    - Longer expiry (60-120 min) for swing strategies
    - Consider market session and volatility patterns
    """)

# Logout button
if st.session_state.authenticated:
    if st.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.session_state.auth_token = None
        st.rerun()

# Auto-refresh indicator
st.caption("🔄 Data refreshes every 30 seconds")
