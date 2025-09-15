"""
Signal Table Component
"""
import streamlit as st
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional

def render_signal_table(
    signals: List[Dict[str, Any]], 
    title: str = "Signals",
    show_actions: bool = False,
    max_rows: Optional[int] = None
) -> None:
    """
    Render a formatted table of trading signals
    
    Args:
        signals: List of signal dictionaries
        title: Table title
        show_actions: Whether to show action buttons
        max_rows: Maximum number of rows to display
    """
    
    if not signals:
        st.info(f"No {title.lower()} available")
        return
    
    # Limit rows if specified
    display_signals = signals[:max_rows] if max_rows else signals
    
    st.subheader(f"📊 {title}")
    
    # Convert to DataFrame for better display
    df_data = []
    for signal in display_signals:
        # Format time
        issued_time = "N/A"
        if signal.get('issued_at'):
            try:
                dt = datetime.fromisoformat(signal['issued_at'].replace('Z', '+00:00'))
                issued_time = dt.strftime("%m/%d %H:%M")
            except (ValueError, TypeError):
                issued_time = str(signal['issued_at'])[:16]
        
        # Format expiry time
        expires_time = "N/A"
        if signal.get('expires_at'):
            try:
                dt = datetime.fromisoformat(signal['expires_at'].replace('Z', '+00:00'))
                expires_time = dt.strftime("%m/%d %H:%M")
            except (ValueError, TypeError):
                expires_time = str(signal['expires_at'])[:16]
        
        df_data.append({
            'Symbol': signal.get('symbol', 'N/A'),
            'Action': signal.get('action', 'N/A'),
            'Price': f"{signal.get('price', 0):.5f}",
            'SL': f"{signal.get('sl', 0):.5f}" if signal.get('sl') else 'N/A',
            'TP': f"{signal.get('tp', 0):.5f}" if signal.get('tp') else 'N/A',
            'Confidence': f"{signal.get('confidence', 0):.2f}",
            'Strategy': signal.get('strategy', 'N/A'),
            'Issued': issued_time,
            'Expires': expires_time,
            'WhatsApp': "✅" if signal.get('sent_to_whatsapp') else "❌",
            'Risk': "🚫" if signal.get('blocked_by_risk') else "✅",
            'ID': signal.get('id', 0)
        })
    
    df = pd.DataFrame(df_data)
    
    # Style the dataframe
    def style_dataframe(df):
        """Apply custom styling to the dataframe"""
        def style_action(val):
            # Handle all MT5 order types with appropriate colors
            if val in ['BUY', 'BUY LIMIT', 'BUY STOP', 'BUY STOP LIMIT']:
                return 'background-color: #d4edda; color: #155724; font-weight: bold'
            elif val in ['SELL', 'SELL LIMIT', 'SELL STOP', 'SELL STOP LIMIT']:
                return 'background-color: #f8d7da; color: #721c24; font-weight: bold'
            return ''
        
        def style_confidence(val):
            try:
                conf = float(val)
                if conf >= 0.8:
                    return 'background-color: #d4edda; color: #155724'
                elif conf >= 0.6:
                    return 'background-color: #fff3cd; color: #856404'
                else:
                    return 'background-color: #f8d7da; color: #721c24'
            except:
                return ''
        
        def style_status(val):
            if val == "✅":
                return 'color: green; font-weight: bold'
            elif val == "❌" or val == "🚫":
                return 'color: red; font-weight: bold'
            return ''
        
        styled = df.style.applymap(style_action, subset=['Action'])
        styled = styled.applymap(style_confidence, subset=['Confidence'])
        styled = styled.applymap(style_status, subset=['WhatsApp', 'Risk'])
        
        return styled
    
    # Display the styled dataframe
    styled_df = style_dataframe(df)
    
    # Remove ID column from display if not needed
    display_columns = [col for col in df.columns if col != 'ID' or show_actions]
    
    st.dataframe(
        styled_df[display_columns] if not show_actions else styled_df,
        use_container_width=True,
        height=min(400, len(df) * 35 + 50)
    )
    
    # Show action buttons if requested
    if show_actions and st.session_state.get('authenticated') and st.session_state.get('user_role') == 'admin':
        st.subheader("🎛️ Signal Actions")
        
        # Select signal for actions
        signal_options = [f"{row['Symbol']} {row['Action']} @ {row['Price']} (ID: {row['ID']})" 
                         for _, row in df.iterrows()]
        
        if signal_options:
            selected_signal_str = st.selectbox(
                "Select Signal for Actions:",
                options=signal_options,
                index=0
            )
            
            # Extract signal ID
            signal_id = int(selected_signal_str.split("ID: ")[1].split(")")[0])
            selected_signal = next((s for s in signals if s.get('id') == signal_id), None)
            
            if selected_signal:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📱 Resend to WhatsApp", use_container_width=True):
                        # This would call the resend API
                        st.info(f"Would resend signal {signal_id} to WhatsApp")
                
                with col2:
                    if st.button("📋 Copy Signal Data", use_container_width=True):
                        signal_text = format_signal_text(selected_signal)
                        st.code(signal_text)
                        st.success("Signal data copied to display")
                
                with col3:
                    if st.button("🔍 View Details", use_container_width=True):
                        st.json(selected_signal)

def render_signal_summary(signals: List[Dict[str, Any]]) -> None:
    """
    Render a summary of signals with key metrics
    
    Args:
        signals: List of signal dictionaries
    """
    
    if not signals:
        st.info("No signals available for summary")
        return
    
    st.subheader("📈 Signal Summary")
    
    # Calculate metrics
    total_signals = len(signals)
    buy_signals = len([s for s in signals if s.get('action', '').startswith('BUY')])
    sell_signals = len([s for s in signals if s.get('action', '').startswith('SELL')])
    blocked_signals = len([s for s in signals if s.get('blocked_by_risk')])
    sent_signals = len([s for s in signals if s.get('sent_to_whatsapp')])
    
    # Average confidence
    confidences = [s.get('confidence', 0) for s in signals if s.get('confidence')]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    # Symbol distribution
    symbol_counts = {}
    for signal in signals:
        symbol = signal.get('symbol', 'Unknown')
        symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
    
    # Strategy distribution
    strategy_counts = {}
    for signal in signals:
        strategy = signal.get('strategy', 'Unknown')
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Signals", total_signals)
        st.metric("Buy/Sell Ratio", f"{buy_signals}/{sell_signals}")
    
    with col2:
        success_rate = (sent_signals / total_signals * 100) if total_signals > 0 else 0
        st.metric("Delivery Rate", f"{success_rate:.1f}%")
        st.metric("Sent to WhatsApp", sent_signals)
    
    with col3:
        block_rate = (blocked_signals / total_signals * 100) if total_signals > 0 else 0
        st.metric("Block Rate", f"{block_rate:.1f}%")
        st.metric("Blocked by Risk", blocked_signals)
    
    with col4:
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
        st.metric("Top Symbol", max(symbol_counts.items(), key=lambda x: x[1])[0] if symbol_counts else "N/A")
    
    # Distribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        if symbol_counts:
            st.subheader("Symbol Distribution")
            chart_data = pd.DataFrame(
                list(symbol_counts.items()),
                columns=['Symbol', 'Count']
            )
            st.bar_chart(chart_data.set_index('Symbol'))
    
    with col2:
        if strategy_counts:
            st.subheader("Strategy Distribution")
            chart_data = pd.DataFrame(
                list(strategy_counts.items()),
                columns=['Strategy', 'Count']
            )
            st.bar_chart(chart_data.set_index('Strategy'))

def render_signal_filters() -> Dict[str, Any]:
    """
    Render signal filter controls and return filter parameters
    
    Returns:
        Dictionary with filter parameters
    """
    
    st.subheader("🔍 Signal Filters")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        symbol_filter = st.selectbox(
            "Symbol",
            options=["ALL", "EURUSD", "GBPUSD", "USDJPY"],
            index=0
        )
    
    with col2:
        action_filter = st.selectbox(
            "Action",
            options=["ALL", "BUY", "SELL", "BUY LIMIT", "SELL LIMIT", "BUY STOP", "SELL STOP", "BUY STOP LIMIT", "SELL STOP LIMIT"],
            index=0
        )
    
    with col3:
        strategy_filter = st.selectbox(
            "Strategy",
            options=["ALL", "ema_rsi", "donchian_atr", "meanrev_bb"],
            index=0
        )
    
    with col4:
        status_filter = st.selectbox(
            "Status",
            options=["ALL", "Sent", "Blocked", "Pending"],
            index=0
        )
    
    return {
        'symbol': symbol_filter if symbol_filter != "ALL" else None,
        'action': action_filter if action_filter != "ALL" else None,
        'strategy': strategy_filter if strategy_filter != "ALL" else None,
        'status': status_filter if status_filter != "ALL" else None
    }

def apply_signal_filters(signals: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Apply filters to signal list
    
    Args:
        signals: List of signal dictionaries
        filters: Filter parameters from render_signal_filters()
    
    Returns:
        Filtered list of signals
    """
    
    filtered_signals = signals.copy()
    
    # Apply symbol filter
    if filters.get('symbol'):
        filtered_signals = [s for s in filtered_signals if s.get('symbol') == filters['symbol']]
    
    # Apply action filter
    if filters.get('action'):
        if filters['action'] in ['BUY', 'SELL']:
            # For backward compatibility, also match specific MT5 order types
            filtered_signals = [s for s in filtered_signals if s.get('action', '').startswith(filters['action'])]
        else:
            # Exact match for specific MT5 order types
            filtered_signals = [s for s in filtered_signals if s.get('action') == filters['action']]
    
    # Apply strategy filter
    if filters.get('strategy'):
        filtered_signals = [s for s in filtered_signals if s.get('strategy') == filters['strategy']]
    
    # Apply status filter
    if filters.get('status'):
        if filters['status'] == 'Sent':
            filtered_signals = [s for s in filtered_signals if s.get('sent_to_whatsapp')]
        elif filters['status'] == 'Blocked':
            filtered_signals = [s for s in filtered_signals if s.get('blocked_by_risk')]
        elif filters['status'] == 'Pending':
            filtered_signals = [s for s in filtered_signals if not s.get('sent_to_whatsapp') and not s.get('blocked_by_risk')]
    
    return filtered_signals

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
    
    text = f"{symbol} {action} @ {price:.5f} | SL {sl_str} | TP {tp_str} | conf {confidence:.2f} | {strategy}"
    
    # Add timing info if available
    if signal.get('issued_at'):
        try:
            dt = datetime.fromisoformat(signal['issued_at'].replace('Z', '+00:00'))
            text += f"\nIssued: {dt.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        except:
            pass
    
    if signal.get('expires_at'):
        try:
            dt = datetime.fromisoformat(signal['expires_at'].replace('Z', '+00:00'))
            text += f"\nExpires: {dt.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        except:
            pass
    
    return text
