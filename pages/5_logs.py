"""
Logs Viewer Page
"""
import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import json

st.set_page_config(page_title="Logs", page_icon="📋", layout="wide")

st.title("📋 System Logs")

# Helper function to call backend API
def call_api(endpoint, method="GET", data=None, token=None):
    """Call backend API"""
    try:
        base_url = "http://0.0.0.0:8000"
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

# Load recent signals for log simulation
@st.cache_data(ttl=30)
def load_recent_signals():
    """Load recent signals to simulate log events"""
    return call_api("/api/signals/recent?limit=100")

# Since we don't have a dedicated logs endpoint, we'll simulate from signals
recent_signals = load_recent_signals()

# Filter and display options
st.subheader("🔍 Log Filters")

col1, col2, col3, col4 = st.columns(4)

with col1:
    log_level = st.selectbox(
        "Log Level",
        options=["ALL", "INFO", "WARNING", "ERROR"],
        index=0
    )

with col2:
    log_source = st.selectbox(
        "Source",
        options=["ALL", "signal_engine", "whatsapp_service", "risk_manager", "api"],
        index=0
    )

with col3:
    time_range = st.selectbox(
        "Time Range",
        options=["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days"],
        index=2
    )

with col4:
    auto_refresh = st.checkbox("Auto Refresh (30s)", value=True)

# Generate simulated log entries from signals
def generate_log_entries(signals):
    """Generate log entries from signal data"""
    logs = []
    
    if not signals:
        return logs
    
    for signal in signals:
        timestamp = datetime.fromisoformat(signal['issued_at'].replace('Z', '+00:00'))
        
        # Signal generated log
        logs.append({
            'timestamp': timestamp,
            'level': 'INFO',
            'source': 'signal_engine',
            'message': f"Signal generated for {signal['symbol']}",
            'details': {
                'event_type': 'signal_generated',
                'symbol': signal['symbol'],
                'action': signal['action'],
                'price': signal['price'],
                'confidence': signal['confidence'],
                'strategy': signal['strategy']
            }
        })
        
        # Risk management log
        if signal.get('blocked_by_risk'):
            logs.append({
                'timestamp': timestamp + timedelta(seconds=1),
                'level': 'WARNING',
                'source': 'risk_manager',
                'message': f"Signal blocked by risk management: {signal.get('risk_reason', 'Unknown reason')}",
                'details': {
                    'event_type': 'risk_block',
                    'symbol': signal['symbol'],
                    'reason': signal.get('risk_reason')
                }
            })
        else:
            # WhatsApp log
            if signal.get('sent_to_whatsapp'):
                logs.append({
                    'timestamp': timestamp + timedelta(seconds=2),
                    'level': 'INFO',
                    'source': 'whatsapp_service',
                    'message': f"WhatsApp message sent for {signal['symbol']} signal",
                    'details': {
                        'event_type': 'whatsapp_sent',
                        'signal_id': signal.get('id'),
                        'symbol': signal['symbol'],
                        'success': True
                    }
                })
            else:
                logs.append({
                    'timestamp': timestamp + timedelta(seconds=2),
                    'level': 'ERROR',
                    'source': 'whatsapp_service',
                    'message': f"Failed to send WhatsApp message for {signal['symbol']} signal",
                    'details': {
                        'event_type': 'whatsapp_failed',
                        'signal_id': signal.get('id'),
                        'symbol': signal['symbol'],
                        'success': False
                    }
                })
    
    # Add some system logs
    now = datetime.utcnow()
    logs.extend([
        {
            'timestamp': now - timedelta(minutes=5),
            'level': 'INFO',
            'source': 'system',
            'message': 'Signal scheduler heartbeat',
            'details': {'event_type': 'system_heartbeat', 'component': 'scheduler'}
        },
        {
            'timestamp': now - timedelta(minutes=10),
            'level': 'INFO',
            'source': 'api',
            'message': 'Health check endpoint accessed',
            'details': {'event_type': 'api_request', 'endpoint': '/api/health'}
        }
    ])
    
    return sorted(logs, key=lambda x: x['timestamp'], reverse=True)

# Generate logs
log_entries = generate_log_entries(recent_signals)

# Apply filters
filtered_logs = log_entries

if log_level != "ALL":
    filtered_logs = [log for log in filtered_logs if log['level'] == log_level]

if log_source != "ALL":
    filtered_logs = [log for log in filtered_logs if log['source'] == log_source]

# Apply time range filter
now = datetime.utcnow()
time_deltas = {
    "Last Hour": timedelta(hours=1),
    "Last 6 Hours": timedelta(hours=6),
    "Last 24 Hours": timedelta(hours=24),
    "Last 7 Days": timedelta(days=7)
}

if time_range in time_deltas:
    cutoff_time = now - time_deltas[time_range]
    filtered_logs = [log for log in filtered_logs if log['timestamp'] >= cutoff_time]

# Display log statistics
st.markdown("---")
st.subheader("📊 Log Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_logs = len(filtered_logs)
    st.metric("Total Events", total_logs)

with col2:
    error_logs = len([log for log in filtered_logs if log['level'] == 'ERROR'])
    st.metric("Errors", error_logs, delta=f"{(error_logs/total_logs*100):.1f}%" if total_logs > 0 else "0%")

with col3:
    warning_logs = len([log for log in filtered_logs if log['level'] == 'WARNING'])
    st.metric("Warnings", warning_logs, delta=f"{(warning_logs/total_logs*100):.1f}%" if total_logs > 0 else "0%")

with col4:
    info_logs = len([log for log in filtered_logs if log['level'] == 'INFO'])
    st.metric("Info", info_logs, delta=f"{(info_logs/total_logs*100):.1f}%" if total_logs > 0 else "0%")

# Log level distribution chart
if filtered_logs:
    level_counts = {}
    for log in filtered_logs:
        level_counts[log['level']] = level_counts.get(log['level'], 0) + 1
    
    import plotly.graph_objects as go
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(level_counts.keys()),
            y=list(level_counts.values()),
            marker_color=['green' if level == 'INFO' else 'orange' if level == 'WARNING' else 'red' for level in level_counts.keys()]
        )
    ])
    
    fig.update_layout(
        title="Log Level Distribution",
        xaxis_title="Log Level",
        yaxis_title="Count",
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Display logs
st.markdown("---")
st.subheader("📋 Event Log")

if filtered_logs:
    # Pagination
    page_size = 50
    total_pages = (len(filtered_logs) + page_size - 1) // page_size
    
    if total_pages > 1:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            page = st.selectbox(
                f"Page (showing {min(page_size, len(filtered_logs))} of {len(filtered_logs)} events)",
                options=list(range(1, total_pages + 1)),
                index=0
            )
        
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, len(filtered_logs))
        page_logs = filtered_logs[start_idx:end_idx]
    else:
        page_logs = filtered_logs[:page_size]
    
    # Display logs
    for i, log in enumerate(page_logs):
        # Determine log color
        if log['level'] == 'ERROR':
            color = "#ffebee"
            icon = "🔴"
        elif log['level'] == 'WARNING':
            color = "#fff3e0"
            icon = "🟡"
        else:
            color = "#e8f5e8"
            icon = "🟢"
        
        with st.container():
            # Main log entry
            time_str = log['timestamp'].strftime("%Y-%m-%d %H:%M:%S UTC")
            
            col1, col2 = st.columns([1, 5])
            
            with col1:
                st.markdown(f"**{icon} {log['level']}**")
                st.caption(time_str)
                st.caption(f"📍 {log['source']}")
            
            with col2:
                st.markdown(f"**{log['message']}**")
                
                # Show details if available
                if log.get('details'):
                    with st.expander("View Details"):
                        st.json(log['details'])
            
            st.markdown("---")

else:
    st.info("No log entries match the current filters")
    st.markdown("""
    **Possible reasons:**
    - No recent system activity
    - Filters are too restrictive
    - System is starting up
    
    Try:
    - Expanding the time range
    - Changing log level to 'ALL'
    - Checking if the backend is running
    """)

# Export functionality
st.markdown("---")
st.subheader("📤 Export Logs")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📄 Export as JSON", use_container_width=True):
        if filtered_logs:
            # Convert timestamps to strings for JSON serialization
            export_logs = []
            for log in filtered_logs:
                export_log = log.copy()
                export_log['timestamp'] = log['timestamp'].isoformat()
                export_logs.append(export_log)
            
            json_str = json.dumps(export_logs, indent=2)
            st.download_button(
                label="💾 Download JSON",
                data=json_str,
                file_name=f"forex_signals_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        else:
            st.warning("No logs to export")

with col2:
    if st.button("📊 Export as CSV", use_container_width=True):
        if filtered_logs:
            # Convert to DataFrame
            df_data = []
            for log in filtered_logs:
                df_data.append({
                    'timestamp': log['timestamp'].isoformat(),
                    'level': log['level'],
                    'source': log['source'],
                    'message': log['message'],
                    'details': json.dumps(log.get('details', {}))
                })
            
            df = pd.DataFrame(df_data)
            csv = df.to_csv(index=False)
            
            st.download_button(
                label="💾 Download CSV",
                data=csv,
                file_name=f"forex_signals_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No logs to export")

with col3:
    if st.button("🔄 Refresh Logs", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Real-time monitoring note
st.markdown("---")
st.info("""
💡 **Note**: This log viewer shows simulated events based on signal data. 
In a production environment, logs would be stored in a dedicated logging system 
with structured log entries, real-time streaming, and advanced search capabilities.
""")

if auto_refresh:
    st.caption("🔄 Auto-refresh enabled - page will update every 30 seconds")
