"""
Fixed Signal Table Component - Addresses critical regressions
"""
import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

def call_api(endpoint: str, method: str = "GET", data: dict = None) -> dict:
    """Make API call with error handling"""
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
            return {"error": f"API returned {response.status_code}"}
            
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def get_signal_status(signal: Dict[str, Any]) -> tuple[str, str]:
    """
    Get signal status with proper logic - fixes critical regression
    
    Returns:
        Tuple of (status_text, status_color)
    """
    # Use result field from Signal model if available
    result = signal.get('result', 'PENDING')
    
    if result in ['WIN', 'LOSS']:
        return f"📊 {result}", "green" if result == 'WIN' else "red"
    elif result == 'EXPIRED':
        return "⏰ Expired", "orange"
    elif result == 'PENDING':
        # Check if still active based on expires_at
        expires_at = signal.get('expires_at')
        if expires_at:
            try:
                # Handle multiple datetime formats from API
                if isinstance(expires_at, str):
                    # Try parsing with different formats
                    if 'T' in expires_at:
                        expires_time = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
                    else:
                        # Handle simple date format
                        expires_time = datetime.strptime(expires_at, '%Y-%m-%d %H:%M:%S')
                        expires_time = expires_time.replace(tzinfo=timezone.utc)
                else:
                    # Already a datetime object
                    expires_time = expires_at
                
                current_time = datetime.now(timezone.utc)
                
                if expires_time > current_time:
                    return "🟢 Active", "green"
                else:
                    return "⏰ Expired", "orange"
            except (ValueError, TypeError, AttributeError):
                # If timestamp parsing fails, default to Active for PENDING signals
                return "🟢 Active", "green"
        else:
            return "🟡 Pending", "blue"
    else:
        # For any unrecognized result, default to Active status to avoid "Unknown"
        return "🟢 Active", "green"

def render_signal_table(
    signals: List[Dict[str, Any]], 
    title: str = "Signals",
    show_details: bool = False,
    max_rows: Optional[int] = None
) -> None:
    """
    Render a functional signal table with restored actions (fixes critical regressions)
    
    Args:
        signals: List of signal dictionaries
        title: Table title
        show_details: Whether to show detailed view with all columns
        max_rows: Maximum number of rows to display
    """
    
    if not signals:
        st.info(f"No {title.lower()} available")
        return
    
    # Limit rows if specified
    display_signals = signals[:max_rows] if max_rows else signals
    
    st.subheader(f"📊 {title}")
    
    # Toggle for simple/detailed view
    view_mode = st.radio(
        "View Mode:",
        ["Simple", "Detailed"],
        horizontal=True,
        index=0 if not show_details else 1,
        key=f"view_mode_{title.replace(' ', '_')}"
    )
    
    if view_mode == "Simple":
        _render_simple_table(display_signals)
    else:
        _render_detailed_table(display_signals)

def _render_simple_table(signals: List[Dict[str, Any]]) -> None:
    """Render simplified table with essential columns only - FIXED VERSION"""
    
    # Create rows using Streamlit columns (more reliable than pandas styling)
    header_cols = st.columns([1, 0.8, 1, 1, 0.8, 0.6, 1.2])
    header_cols[0].write("**Symbol**")
    header_cols[1].write("**Signal**")
    header_cols[2].write("**Price**")
    header_cols[3].write("**Confidence**")
    header_cols[4].write("**Status**")
    header_cols[5].write("**Time**")
    header_cols[6].write("**Actions**")
    
    st.divider()
    
    for signal in signals:
        cols = st.columns([1, 0.8, 1, 1, 0.8, 0.6, 1.2])
        
        # Symbol
        cols[0].write(signal.get('symbol', 'N/A'))
        
        # Signal with proper Streamlit-native coloring
        action = signal.get('action', 'N/A')
        if action == 'BUY':
            cols[1].success(f"🔼 {action}")
        elif action == 'SELL':
            cols[1].error(f"🔻 {action}")
        else:
            cols[1].write(action)
        
        # Price
        cols[2].write(f"{signal.get('price', 0):.5f}")
        
        # Confidence with proper color coding
        confidence = signal.get('confidence', 0)
        conf_text = f"{confidence:.0%}"
        if confidence >= 0.8:
            cols[3].success(conf_text)
        elif confidence >= 0.7:
            cols[3].warning(conf_text)
        else:
            cols[3].error(conf_text)
        
        # Status using FIXED logic
        status_text, status_color = get_signal_status(signal)
        if status_color == "green":
            cols[4].success(status_text)
        elif status_color == "red":
            cols[4].error(status_text)
        elif status_color == "orange":
            cols[4].warning(status_text)
        else:
            cols[4].info(status_text)
        
        # Time
        try:
            issued_time = datetime.fromisoformat(signal['issued_at'].replace('Z', '+00:00'))
            time_str = issued_time.strftime("%H:%M")
        except:
            time_str = "N/A"
        cols[5].write(time_str)
        
        # Actions - RESTORED critical functionality
        with cols[6]:
            action_cols = st.columns(2)
            
            # Test message button
            with action_cols[0]:
                if st.button("📋", key=f"test_{signal.get('id', 0)}", 
                           help="Test message", use_container_width=True):
                    _handle_test_signal(signal)
            
            # Resend button (preserved structure)
            with action_cols[1]:
                if st.button("📱", key=f"resend_{signal.get('id', 0)}", 
                           help="Resend signal", use_container_width=True):
                    _handle_resend_signal(signal)
    
    # Simple summary stats with FIXED calculation
    active_count = sum(1 for s in signals if get_signal_status(s)[0].startswith('🟢'))
    buy_count = sum(1 for s in signals if s.get('action') == 'BUY')
    sell_count = sum(1 for s in signals if s.get('action') == 'SELL')
    
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Active Signals", active_count)
    with col2:
        st.metric("Buy Signals", buy_count)
    with col3:
        st.metric("Sell Signals", sell_count)

def _render_detailed_table(signals: List[Dict[str, Any]]) -> None:
    """Render detailed table with all available columns - FIXED VERSION"""
    
    # Use st.data_editor for better reliability
    table_data = []
    for signal in signals:
        # Format time
        try:
            issued_time = datetime.fromisoformat(signal['issued_at'].replace('Z', '+00:00'))
            time_str = issued_time.strftime("%m/%d %H:%M")
        except:
            time_str = "N/A"
        
        # Format expiry time
        try:
            expires_time = datetime.fromisoformat(signal['expires_at'].replace('Z', '+00:00'))
            expires_str = expires_time.strftime("%m/%d %H:%M")
        except:
            expires_str = "N/A"
        
        # Get proper status
        status_text, _ = get_signal_status(signal)
        
        table_data.append({
            'Time': time_str,
            'Symbol': signal.get('symbol', 'N/A'),
            'Signal': signal.get('action', 'N/A'),
            'Price': f"{signal.get('price', 0):.5f}",
            'Stop Loss': f"{signal.get('sl', 0):.5f}" if signal.get('sl') else 'N/A',
            'Take Profit': f"{signal.get('tp', 0):.5f}" if signal.get('tp') else 'N/A',
            'Confidence': f"{signal.get('confidence', 0):.0%}",
            'Strategy': signal.get('strategy', 'N/A'),
            'Expires': expires_str,
            'Status': status_text,
            'Sent': "✅" if signal.get('sent_to_whatsapp') else "❌",
            'Blocked': "🚫" if signal.get('blocked_by_risk') else "✅"
        })
    
    df = pd.DataFrame(table_data)
    
    # Configure columns properly
    column_config = {
        'Time': st.column_config.TextColumn('Time', width='small'),
        'Symbol': st.column_config.TextColumn('Symbol', width='small'),
        'Signal': st.column_config.TextColumn('Signal', width='small'),
        'Price': st.column_config.TextColumn('Price', width='medium'),
        'Stop Loss': st.column_config.TextColumn('SL', width='medium'),
        'Take Profit': st.column_config.TextColumn('TP', width='medium'),
        'Confidence': st.column_config.TextColumn('Conf', width='small'),
        'Strategy': st.column_config.TextColumn('Strategy', width='medium'),
        'Expires': st.column_config.TextColumn('Expires', width='small'),
        'Status': st.column_config.TextColumn('Status', width='small'),
        'Sent': st.column_config.TextColumn('Sent', width='small'),
        'Blocked': st.column_config.TextColumn('Risk', width='small')
    }
    
    st.dataframe(df, column_config=column_config, use_container_width=True, hide_index=True, height=400)
    
    # Add actions for detailed view
    _render_action_section(signals)

def _render_action_section(signals: List[Dict[str, Any]]) -> None:
    """Render action section for detailed signals"""
    
    if not signals:
        return
        
    st.subheader("🎛️ Signal Actions")
    
    # Select signal
    signal_options = [
        f"{signal.get('symbol', 'N/A')} {signal.get('action', 'N/A')} @ {signal.get('price', 0):.5f} (ID: {signal.get('id', 0)})" 
        for signal in signals
    ]
    
    if signal_options:
        selected_signal_str = st.selectbox(
            "Select Signal:",
            options=signal_options,
            key="detailed_signal_select"
        )
        
        # Extract signal ID
        try:
            signal_id = int(selected_signal_str.split("ID: ")[1].split(")")[0])
            selected_signal = next((s for s in signals if s.get('id') == signal_id), None)
        except:
            selected_signal = None
        
        if selected_signal:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📋 Test Message", key="test_detailed"):
                    _handle_test_signal(selected_signal)
            
            with col2:
                if st.button("📱 Resend Signal", key="resend_detailed"):
                    _handle_resend_signal(selected_signal)
            
            with col3:
                if st.button("🔍 View Details", key="details_detailed"):
                    st.json(selected_signal)

def _handle_test_signal(signal: Dict[str, Any]) -> None:
    """Handle test signal functionality - RESTORED"""
    signal_id = signal.get('id', 0)
    
    # Show formatted signal data
    signal_text = format_signal_text(signal)
    
    with st.expander(f"📋 Test Signal {signal_id}", expanded=True):
        st.code(signal_text, language="text")
        
        # Test API call
        result = call_api(f"/api/signals/{signal_id}/test", method="POST")
        
        if "error" in result:
            st.warning(f"⚠️ Test result: {result['error']}")
        else:
            st.success("✅ Test message generated successfully!")

def _handle_resend_signal(signal: Dict[str, Any]) -> None:
    """Handle resend signal functionality - RESTORED"""
    signal_id = signal.get('id', 0)
    
    # Call resend API
    result = call_api(f"/api/signals/{signal_id}/resend", method="POST")
    
    if "error" in result:
        st.error(f"❌ Failed to resend: {result['error']}")
        st.info("💡 Note: WhatsApp integration may not be configured")
    else:
        st.success(f"✅ Signal {signal_id} resent successfully!")
        # Auto-refresh
        st.rerun()

def render_signal_summary(signals: List[Dict[str, Any]]) -> None:
    """
    Render signal summary with FIXED metrics calculation
    
    Args:
        signals: List of signal dictionaries
    """
    
    if not signals:
        st.info("No signals available for summary")
        return
    
    st.subheader("📊 Signal Summary")
    
    # FIXED metrics calculation - no more broken string searches
    total_signals = len(signals)
    buy_signals = sum(1 for s in signals if s.get('action') == 'BUY')
    sell_signals = sum(1 for s in signals if s.get('action') == 'SELL')
    
    # Use proper status logic
    active_signals = sum(1 for s in signals if get_signal_status(s)[0].startswith('🟢'))
    
    # Success metrics
    sent_signals = sum(1 for s in signals if s.get('sent_to_whatsapp', False))
    blocked_signals = sum(1 for s in signals if s.get('blocked_by_risk', False))
    
    # Average confidence
    confidences = [s.get('confidence', 0) for s in signals if s.get('confidence') is not None]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Signals", total_signals)
    
    with col2:
        st.metric("Buy Signals", buy_signals)
        st.metric("Sell Signals", sell_signals)
    
    with col3:
        st.metric("Active Signals", active_signals)
        success_rate = (sent_signals / total_signals * 100) if total_signals > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    with col4:
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        block_rate = (blocked_signals / total_signals * 100) if total_signals > 0 else 0
        st.metric("Block Rate", f"{block_rate:.1f}%")

def format_signal_text(signal: Dict[str, Any]) -> str:
    """
    Format signal data as text for copying/sharing
    
    Args:
        signal: Signal dictionary
    
    Returns:
        Formatted signal text
    """
    
    symbol = signal.get('symbol', 'N/A')
    action = signal.get('action', 'N/A')
    price = signal.get('price', 0)
    sl = signal.get('sl')
    tp = signal.get('tp')
    confidence = signal.get('confidence', 0)
    strategy = signal.get('strategy', 'N/A')
    
    sl_str = f"{sl:.5f}" if sl else 'N/A'
    tp_str = f"{tp:.5f}" if tp else 'N/A'
    
    text = f"📊 {symbol} {action} @ {price:.5f}\n"
    text += f"Stop Loss: {sl_str}\n"
    text += f"Take Profit: {tp_str}\n"
    text += f"Confidence: {confidence:.1%}\n"
    text += f"Strategy: {strategy}\n"
    
    # Add timing info if available
    if signal.get('issued_at'):
        try:
            dt = datetime.fromisoformat(signal['issued_at'].replace('Z', '+00:00'))
            text += f"Issued: {dt.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        except:
            pass
    
    if signal.get('expires_at'):
        try:
            dt = datetime.fromisoformat(signal['expires_at'].replace('Z', '+00:00'))
            text += f"Expires: {dt.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        except:
            pass
    
    # Add status
    status_text, _ = get_signal_status(signal)
    text += f"Status: {status_text}"
    
    return text