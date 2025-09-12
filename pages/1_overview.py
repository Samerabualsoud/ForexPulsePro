"""
Overview Page - Main dashboard with signal overview and quick actions
"""
import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Overview", page_icon="📈", layout="wide")

st.title("📈 Overview Dashboard")

# Helper function to call backend API
def call_api(endpoint, method="GET", data=None):
    """Call backend API"""
    try:
        base_url = "http://localhost:8000"
        url = f"{base_url}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
        return None

# Load data
@st.cache_data(ttl=60)  # Cache for 1 minute
def load_recent_signals():
    """Load recent signals from API"""
    return call_api("/api/signals/recent?limit=20")

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_risk_status():
    """Load risk status from API"""
    return call_api("/api/risk/status")

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_success_rate():
    """Load success rate statistics from API"""
    return call_api("/api/signals/success-rate?days=30")

# Load data
recent_signals = load_recent_signals()
risk_status = load_risk_status()
success_rate = load_success_rate()

# Top metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    if risk_status:
        kill_switch_status = "🔴 ON" if risk_status.get('kill_switch_enabled') else "🟢 OFF"
        st.metric(
            "Kill Switch", 
            kill_switch_status,
            delta="BLOCKED" if risk_status.get('kill_switch_enabled') else "ACTIVE"
        )
    else:
        st.metric("Kill Switch", "❓ Unknown")

with col2:
    if recent_signals:
        last_signal_time = "Never"
        if recent_signals and len(recent_signals) > 0:
            last_issued = recent_signals[0].get('issued_at')
            if last_issued:
                dt = datetime.fromisoformat(last_issued.replace('Z', '+00:00'))
                last_signal_time = dt.strftime("%H:%M UTC")
        
        st.metric("Last Signal", last_signal_time)
    else:
        st.metric("Last Signal", "❓ Unknown")

with col3:
    if recent_signals:
        today_count = 0
        today = datetime.now().date()
        for signal in recent_signals:
            signal_date = datetime.fromisoformat(signal['issued_at'].replace('Z', '+00:00')).date()
            if signal_date == today:
                today_count += 1
        
        st.metric("Today's Signals", today_count)
    else:
        st.metric("Today's Signals", "❓")

with col4:
    if success_rate:
        success_pct = success_rate.get('success_rate', 0)
        total_signals = success_rate.get('total_signals', 0)
        
        st.metric(
            "Success Rate (30d)", 
            f"{success_pct}%",
            delta=f"{total_signals} signals"
        )
    else:
        st.metric("Success Rate", "❓")

st.markdown("---")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Recent Signals")
    
    if recent_signals and len(recent_signals) > 0:
        # Convert to DataFrame for display
        df_data = []
        for signal in recent_signals:
            df_data.append({
                'Time': datetime.fromisoformat(signal['issued_at'].replace('Z', '+00:00')).strftime("%H:%M:%S"),
                'Symbol': signal.get('symbol', 'N/A'),
                'Action': signal.get('action', 'N/A'),
                'Price': f"{signal.get('price', 0):.5f}",
                'SL': f"{signal.get('sl', 0):.5f}" if signal.get('sl') else 'N/A',
                'TP': f"{signal.get('tp', 0):.5f}" if signal.get('tp') else 'N/A',
                'Confidence': f"{signal.get('confidence', 0):.2f}",
                'Strategy': signal.get('strategy', 'N/A'),
                'Result': signal.get('result', 'PENDING'),
                'WhatsApp': "✅" if signal.get('sent_to_whatsapp') else "❌",
                'Blocked': "🚫" if signal.get('blocked_by_risk') else "✅"
            })
        
        df = pd.DataFrame(df_data)
        
        # Style the dataframe
        def style_action(val):
            if val == 'BUY':
                return 'background-color: #d4edda; color: #155724'
            elif val == 'SELL':
                return 'background-color: #f8d7da; color: #721c24'
            return ''
        
        styled_df = df.style.map(style_action, subset=['Action'])
        st.dataframe(styled_df, width='stretch', hide_index=True)
        
        # Quick action buttons for recent signals
        st.subheader("Quick Actions")
        
        if st.button("🔄 Refresh Data", width='stretch'):
            st.cache_data.clear()
            st.rerun()
        
        # WhatsApp actions for the latest signal
        if len(recent_signals) > 0:
            latest_signal = recent_signals[0]
            col_a, col_b = st.columns(2)
            
            with col_a:
                if st.button(f"📱 Resend Latest to WhatsApp", width='stretch'):
                    # This would require authentication - simplified for demo
                    st.info("Feature requires admin authentication")
            
            with col_b:
                if st.button("🧪 Test WhatsApp", width='stretch'):
                    st.info("Feature requires admin authentication")
    
    else:
        st.info("No recent signals found. The system may be starting up or no signals have been generated yet.")
        
        if st.button("🔄 Refresh", width='stretch'):
            st.cache_data.clear()
            st.rerun()

with col2:
    st.subheader("System Status")
    
    # Risk status
    if risk_status:
        st.metric("Daily Loss Limit", f"${risk_status.get('daily_loss_limit', 0):.0f}")
        st.metric("Volatility Guard", "🟢 ON" if risk_status.get('volatility_guard_enabled') else "🔴 OFF")
        
        # Progress bar for daily loss
        daily_loss = risk_status.get('current_daily_loss', 0)
        loss_limit = risk_status.get('daily_loss_limit', 1000)
        if loss_limit > 0:
            progress = min(daily_loss / loss_limit, 1.0)
            st.progress(progress)
            st.caption(f"Daily Loss: ${daily_loss:.0f} / ${loss_limit:.0f}")
    
    st.markdown("---")
    
    # Symbol performance today
    st.subheader("Today's Activity")
    
    if recent_signals:
        symbol_counts = {}
        today = datetime.now().date()
        
        for signal in recent_signals:
            signal_date = datetime.fromisoformat(signal['issued_at'].replace('Z', '+00:00')).date()
            if signal_date == today:
                symbol = signal.get('symbol', 'Unknown')
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        
        if symbol_counts:
            for symbol, count in symbol_counts.items():
                st.metric(f"{symbol} Signals", count)
        else:
            st.info("No signals today")
    
    st.markdown("---")
    
    # Recent activity feed
    st.subheader("Activity Feed")
    
    if recent_signals:
        for i, signal in enumerate(recent_signals[:5]):
            time_str = datetime.fromisoformat(signal['issued_at'].replace('Z', '+00:00')).strftime("%H:%M")
            action_color = "🟢" if signal.get('action') == 'BUY' else "🔴" if signal.get('action') == 'SELL' else "⚪"
            
            with st.container():
                st.write(f"{action_color} **{signal.get('symbol')}** {signal.get('action')} @ {signal.get('price', 0):.5f}")
                st.caption(f"{time_str} • {signal.get('strategy')} • Conf: {signal.get('confidence', 0):.2f}")
                
                if i < 4:  # Don't show separator for last item
                    st.markdown("---")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.caption("🔄 Auto-refresh: 60 seconds")

with col2:
    st.caption("📊 Signal Engine: Active")

with col3:
    st.caption(f"📅 Last Updated: {datetime.now().strftime('%H:%M:%S')}")
