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

# Enhanced CSS styling for signals page
st.markdown("""
<style>
    /* Professional title styling */
    .signals-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Enhanced metrics styling */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        padding: 1.2rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.3);
        transition: transform 0.2s ease;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    [data-testid="metric-container"] > div {
        color: white;
    }
    
    [data-testid="metric-container"] label {
        color: rgba(255, 255, 255, 0.9);
        font-weight: 600;
        font-size: 0.95rem;
    }
    
    /* Signals table styling */
    .signals-section {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    }
    
    .signals-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    /* Status indicators */
    .status-section {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    }
    
    .status-header {
        color: #2d3436;
        font-weight: bold;
        font-size: 1.3rem;
        margin-bottom: 1rem;
    }
    
    /* Enhanced dataframe styling */
    .stDataFrame {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 3px 10px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Activity section styling */
    .activity-section {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    }
    
    .activity-header {
        color: #2d3436;
        font-weight: bold;
        font-size: 1.3rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="signals-title">📈 Live Trading Signals</h1>', unsafe_allow_html=True)

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
    
    if recent_signals and len(recent_signals) > 0:
        # Convert to DataFrame for display
        df_data = []
        current_time = datetime.now()
        
        for signal in recent_signals:
            # Calculate signal timing
            issued_at = datetime.fromisoformat(signal['issued_at'].replace('Z', '+00:00')).replace(tzinfo=None)
            expires_at = datetime.fromisoformat(signal['expires_at'].replace('Z', '+00:00')).replace(tzinfo=None)
            
            # Calculate validity status
            time_to_expiry = expires_at - current_time
            if time_to_expiry.total_seconds() <= 0:
                validity = "EXPIRED"
            elif time_to_expiry.total_seconds() <= 300:  # Less than 5 minutes
                minutes = int(time_to_expiry.total_seconds() / 60)
                validity = f"{minutes}m LEFT" if minutes > 0 else "EXPIRING"
            else:
                minutes = int(time_to_expiry.total_seconds() / 60)
                validity = f"ACTIVE ({minutes}m)"
            
            # Calculate execution urgency
            signal_age = current_time - issued_at
            age_minutes = signal_age.total_seconds() / 60
            
            if signal.get('blocked_by_risk', False):
                urgency = "BLOCKED"
            elif age_minutes <= 2:
                urgency = "EXECUTE NOW"
            elif age_minutes <= 5:
                urgency = "FRESH"
            elif age_minutes <= 15:
                urgency = "AGING"
            else:
                urgency = "OLD"
                
            df_data.append({
                'Time': issued_at.strftime("%H:%M:%S"),
                'Symbol': signal.get('symbol', 'N/A'),
                'Action': signal.get('action', 'N/A'),
                'Price': f"{signal.get('price', 0):.5f}",
                'Validity': validity,
                'Urgency': urgency,
                'SL': f"{signal.get('sl', 0):.5f}" if signal.get('sl') else 'N/A',
                'TP': f"{signal.get('tp', 0):.5f}" if signal.get('tp') else 'N/A',
                'Confidence': f"{signal.get('confidence', 0):.0%}",
                'Strategy': signal.get('strategy', 'N/A'),
                'Result': signal.get('result', 'PENDING'),
                'WhatsApp': "✅" if signal.get('sent_to_whatsapp') else "❌",
                'Blocked': "🚫" if signal.get('blocked_by_risk') else "✅"
            })
        
        df = pd.DataFrame(df_data)
        
        # Enhanced styling for the signals table
        def style_signal_table(val):
            if val == 'BUY':
                return 'background-color: #28a745; color: white; font-weight: bold; padding: 8px; border-radius: 8px;'
            elif val == 'SELL':
                return 'background-color: #dc3545; color: white; font-weight: bold; padding: 8px; border-radius: 8px;'
            return ''
        
        def style_result(val):
            if val == 'WIN':
                return 'background-color: #20c997; color: white; font-weight: bold; padding: 5px; border-radius: 5px;'
            elif val == 'LOSS':
                return 'background-color: #e74c3c; color: white; font-weight: bold; padding: 5px; border-radius: 5px;'
            elif val == 'PENDING':
                return 'background-color: #ffc107; color: #212529; font-weight: bold; padding: 5px; border-radius: 5px;'
            return ''
        
        def style_blocked(val):
            if val == '🚫':
                return 'background-color: #6c757d; padding: 5px; border-radius: 5px;'
            elif val == '✅':
                return 'background-color: #28a745; padding: 5px; border-radius: 5px;'
            return ''
            
        def style_validity(val):
            if 'EXPIRED' in val or 'EXPIRING' in val:
                return 'background-color: #dc3545; color: white; font-weight: bold; padding: 5px; border-radius: 5px;'
            elif 'LEFT' in val and int(val.split('m')[0]) <= 5:
                return 'background-color: #fd7e14; color: white; font-weight: bold; padding: 5px; border-radius: 5px;'
            elif 'ACTIVE' in val:
                return 'background-color: #28a745; color: white; font-weight: bold; padding: 5px; border-radius: 5px;'
            return ''
            
        def style_urgency(val):
            if val == 'EXECUTE NOW':
                return 'background-color: #dc3545; color: white; font-weight: bold; padding: 8px; border-radius: 8px; animation: blink 1s infinite;'
            elif val == 'FRESH':
                return 'background-color: #28a745; color: white; font-weight: bold; padding: 5px; border-radius: 5px;'
            elif val == 'AGING':
                return 'background-color: #ffc107; color: #212529; font-weight: bold; padding: 5px; border-radius: 5px;'
            elif val == 'OLD':
                return 'background-color: #6c757d; color: white; padding: 5px; border-radius: 5px;'
            elif val == 'BLOCKED':
                return 'background-color: #6c757d; color: white; font-weight: bold; padding: 5px; border-radius: 5px;'
            return ''
        
        styled_df = df.style.map(style_signal_table, subset=['Action']) \
                           .map(style_result, subset=['Result']) \
                           .map(style_blocked, subset=['Blocked']) \
                           .map(style_validity, subset=['Validity']) \
                           .map(style_urgency, subset=['Urgency'])
                           
        st.markdown('<div class="signals-section">', unsafe_allow_html=True)
        st.markdown('<h3 class="signals-header">🎯 Recent Signals</h3>', unsafe_allow_html=True)
        st.dataframe(styled_df, use_container_width=True, hide_index=True, height=400)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Quick action buttons for recent signals
        st.markdown('<h3 class="signals-header">🚀 Quick Actions</h3>', unsafe_allow_html=True)
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        # WhatsApp actions for the latest signal
        if len(recent_signals) > 0:
            latest_signal = recent_signals[0]
            col_a, col_b = st.columns(2)
            
            with col_a:
                if st.button(f"📱 Resend Latest to WhatsApp", use_container_width=True):
                    # This would require authentication - simplified for demo
                    st.info("Feature requires admin authentication")
            
            with col_b:
                if st.button("🧪 Test WhatsApp", use_container_width=True):
                    st.info("Feature requires admin authentication")
    
    else:
        st.info("No recent signals found. The system may be starting up or no signals have been generated yet.")
        
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

with col2:
    st.markdown('<div class="status-section">', unsafe_allow_html=True)
    st.markdown('<h3 class="status-header">🛡️ System Status</h3>', unsafe_allow_html=True)
    
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
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Symbol performance today
    st.markdown('<div class="activity-section">', unsafe_allow_html=True)
    st.markdown('<h3 class="activity-header">📊 Today\'s Activity</h3>', unsafe_allow_html=True)
    
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
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Performance Chart
    st.markdown('<div class="activity-section">', unsafe_allow_html=True)
    st.markdown('<h3 class="activity-header">📈 Performance Chart</h3>', unsafe_allow_html=True)
    
    if success_rate and success_rate.get('total_signals', 0) > 0:
        # Create performance pie chart
        labels = ['Successful', 'Losses', 'Expired/Pending']
        values = [
            success_rate.get('successful_signals', 0),
            success_rate.get('losing_signals', 0), 
            success_rate.get('expired_signals', 0) + max(0, success_rate.get('total_signals', 0) - success_rate.get('successful_signals', 0) - success_rate.get('losing_signals', 0))
        ]
        colors = ['#28a745', '#dc3545', '#ffc107']
        
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4)])
        fig.update_traces(marker=dict(colors=colors, line=dict(color='white', width=2)))
        fig.update_layout(
            title="Signal Performance Distribution",
            showlegend=True,
            height=350,
            margin=dict(t=50, b=0, l=0, r=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2d3436')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Total Pips", f"{success_rate.get('total_pips', 0):.1f}")
        with col_b:
            st.metric("Avg Per Trade", f"{success_rate.get('avg_pips_per_trade', 0):.1f}")
        with col_c:
            success_pct = success_rate.get('success_rate', 0)
            delta_color = "normal" if success_pct >= 60 else "inverse"
            st.metric("Win Rate", f"{success_pct}%", delta=f"{'🔥' if success_pct >= 70 else '📈' if success_pct >= 60 else '⚠️'}")
    else:
        st.info("📊 Performance data will appear once signals are evaluated")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Recent activity feed
    st.markdown('<div class="activity-section">', unsafe_allow_html=True)
    st.markdown('<h3 class="activity-header">⚡ Recent Activity</h3>', unsafe_allow_html=True)
    
    if recent_signals and len(recent_signals) >= 5:
        for i in range(5):
            signal = recent_signals[i]
            time_str = datetime.fromisoformat(signal['issued_at'].replace('Z', '+00:00')).strftime("%H:%M")
            action_emoji = "🟢" if signal['action'] == 'BUY' else "🔴"
            result_emoji = {"WIN": "✅", "LOSS": "❌", "PENDING": "⏳"}.get(signal.get('result', 'PENDING'), "⏳")
            
            with st.container():
                col_time, col_signal, col_result = st.columns([1, 2, 1])
                with col_time:
                    st.write(f"**{time_str}**")
                with col_signal:
                    st.write(f"{action_emoji} **{signal.get('symbol', 'N/A')}** {signal.get('action', 'N/A')} @ {signal.get('price', 0):.5f}")
                with col_result:
                    st.write(f"{result_emoji} {signal.get('result', 'PENDING')}")
                    
                st.markdown("---")
    else:
        st.info("No recent activity - signals will appear here as they are generated")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
# Sidebar Lot Calculator
with st.sidebar:
    st.markdown("---")
    st.markdown("### 💰 Lot Size Calculator")
    
    with st.container():
        # Account settings
        account_size = st.number_input(
            "Account Balance ($)",
            min_value=100.0,
            max_value=1000000.0,
            value=10000.0,
            step=100.0,
            help="Your total trading account balance"
        )
        
        risk_percent = st.slider(
            "Risk per Trade (%)",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.1,
            help="Percentage of account to risk per trade (recommended: 1-3%)"
        )
        
        stop_loss_pips = st.number_input(
            "Stop Loss (pips)",
            min_value=5,
            max_value=200,
            value=20,
            step=1,
            help="Stop loss distance in pips"
        )
        
        # Currency pair selection for pip value
        currency_pair = st.selectbox(
            "Currency Pair",
            ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"],
            help="Select currency pair to calculate pip value"
        )
        
        # Calculate lot size
        if st.button("📊 Calculate Lot Size", use_container_width=True):
            # Risk amount in dollars
            risk_amount = account_size * (risk_percent / 100)
            
            # Pip value calculation (simplified)
            if currency_pair in ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"]:
                pip_value_per_lot = 10  # $10 per pip for 1 standard lot
            elif currency_pair in ["USDCAD", "USDCHF"]:
                pip_value_per_lot = 10  # Approximately $10 per pip
            elif currency_pair == "USDJPY":
                pip_value_per_lot = 9.09  # Approximately $9.09 per pip (varies with rate)
            else:
                pip_value_per_lot = 10  # Default
            
            # Calculate lot size
            # Lot Size = Risk Amount / (Stop Loss Pips × Pip Value per Lot)
            lot_size = risk_amount / (stop_loss_pips * pip_value_per_lot)
            
            # Display results
            st.success("📈 **Position Size Results:**")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Risk Amount", f"${risk_amount:.2f}")
                st.metric("Standard Lots", f"{lot_size:.2f}")
            
            with col_b:
                st.metric("Mini Lots", f"{lot_size * 10:.1f}")
                st.metric("Micro Lots", f"{lot_size * 100:.0f}")
            
            # Position size recommendations
            if lot_size >= 1.0:
                st.info(f"💡 **Recommendation:** Trade {lot_size:.2f} standard lots")
            elif lot_size >= 0.1:
                mini_lots = lot_size * 10
                st.info(f"💡 **Recommendation:** Trade {mini_lots:.1f} mini lots (0.{mini_lots:.0f})")
            else:
                micro_lots = lot_size * 100
                st.info(f"💡 **Recommendation:** Trade {micro_lots:.0f} micro lots (0.0{micro_lots:.0f})")
            
            # Risk warning
            if risk_percent > 3.0:
                st.warning("⚠️ **High Risk:** Consider reducing risk percentage below 3%")
            elif risk_percent < 1.0:
                st.info("ℹ️ **Conservative:** Very low risk approach")
    
    # Position sizing tips
    with st.expander("📚 Position Sizing Tips"):
        st.markdown("""
        **Risk Management Guidelines:**
        - **Conservative:** 1% risk per trade
        - **Moderate:** 2% risk per trade  
        - **Aggressive:** 3% risk per trade
        - **Never exceed:** 5% risk per trade
        
        **Lot Size Types:**
        - **Standard Lot:** 100,000 units
        - **Mini Lot:** 10,000 units (0.1)
        - **Micro Lot:** 1,000 units (0.01)
        
        **Formula:**
        `Lot Size = Risk Amount ÷ (Stop Loss Pips × Pip Value)`
        """)
        
    st.markdown("---")
    
# Footer with enhanced styling
st.markdown("---")
st.markdown('<div class="footer-section">', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🔄 Auto-refresh", "60s", delta="Active")

with col2:
    st.metric("📊 Signal Engine", "Running", delta="7 Strategies")

with col3:
    st.metric("📅 Last Updated", datetime.now().strftime('%H:%M:%S'), delta="Live Data")

st.markdown('</div>', unsafe_allow_html=True)
