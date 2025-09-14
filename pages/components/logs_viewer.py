"""
Logs Viewer Component
"""
import streamlit as st
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent.parent / "utils"))
from timezone_utils import format_saudi_time_full, to_saudi_time, time_ago_saudi

def render_logs_viewer(
    logs: List[Dict[str, Any]],
    show_filters: bool = True,
    show_export: bool = True,
    max_display: int = 100,
    auto_refresh: bool = False
) -> None:
    """
    Render comprehensive logs viewer component
    
    Args:
        logs: List of log entry dictionaries
        show_filters: Whether to show filter controls
        show_export: Whether to show export options
        max_display: Maximum number of logs to display
        auto_refresh: Whether to enable auto-refresh
    """
    
    st.subheader("📋 System Logs")
    
    if not logs:
        st.info("No log entries available")
        return
    
    # Apply filters if enabled
    if show_filters:
        filtered_logs = render_log_filters(logs)
    else:
        filtered_logs = logs
    
    # Display log statistics
    render_log_statistics(filtered_logs)
    
    # Display logs
    render_log_entries(filtered_logs[:max_display])
    
    # Export options
    if show_export:
        render_log_export(filtered_logs)
    
    # Auto-refresh indicator
    if auto_refresh:
        st.caption("🔄 Auto-refresh enabled")

def render_log_filters(logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Render log filter controls and return filtered logs
    
    Args:
        logs: List of log entries
    
    Returns:
        Filtered list of log entries
    """
    
    st.subheader("🔍 Log Filters")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Extract unique values for filter options
    log_levels = sorted(list(set([log.get('level', 'INFO') for log in logs])))
    log_sources = sorted(list(set([log.get('source', 'unknown') for log in logs])))
    
    with col1:
        level_filter = st.selectbox(
            "Log Level",
            options=["ALL"] + log_levels,
            index=0
        )
    
    with col2:
        source_filter = st.selectbox(
            "Source",
            options=["ALL"] + log_sources,
            index=0
        )
    
    with col3:
        time_range = st.selectbox(
            "Time Range",
            options=["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days", "All"],
            index=2
        )
    
    with col4:
        search_term = st.text_input(
            "Search Message",
            placeholder="Enter search term..."
        )
    
    # Apply filters
    filtered_logs = logs.copy()
    
    # Level filter
    if level_filter != "ALL":
        filtered_logs = [log for log in filtered_logs if log.get('level') == level_filter]
    
    # Source filter
    if source_filter != "ALL":
        filtered_logs = [log for log in filtered_logs if log.get('source') == source_filter]
    
    # Time range filter
    if time_range != "All":
        now = datetime.utcnow()
        time_deltas = {
            "Last Hour": timedelta(hours=1),
            "Last 6 Hours": timedelta(hours=6),
            "Last 24 Hours": timedelta(hours=24),
            "Last 7 Days": timedelta(days=7)
        }
        
        if time_range in time_deltas:
            cutoff_time = now - time_deltas[time_range]
            filtered_logs = [
                log for log in filtered_logs
                if log.get('timestamp') and 
                (isinstance(log['timestamp'], datetime) and log['timestamp'] >= cutoff_time or
                 isinstance(log['timestamp'], str) and datetime.fromisoformat(log['timestamp'].replace('Z', '+00:00')) >= cutoff_time)
            ]
    
    # Search filter
    if search_term:
        search_term_lower = search_term.lower()
        filtered_logs = [
            log for log in filtered_logs
            if search_term_lower in log.get('message', '').lower()
        ]
    
    return filtered_logs

def render_log_statistics(logs: List[Dict[str, Any]]) -> None:
    """
    Render log statistics summary
    
    Args:
        logs: List of log entries
    """
    
    if not logs:
        return
    
    st.subheader("📊 Log Statistics")
    
    # Calculate statistics
    total_logs = len(logs)
    error_count = len([log for log in logs if log.get('level') == 'ERROR'])
    warning_count = len([log for log in logs if log.get('level') == 'WARNING'])
    info_count = len([log for log in logs if log.get('level') == 'INFO'])
    
    # Recent activity (last hour)
    now = datetime.utcnow()
    one_hour_ago = now - timedelta(hours=1)
    recent_logs = [
        log for log in logs
        if log.get('timestamp') and 
        (isinstance(log['timestamp'], datetime) and log['timestamp'] >= one_hour_ago or
         isinstance(log['timestamp'], str) and datetime.fromisoformat(log['timestamp'].replace('Z', '+00:00')) >= one_hour_ago)
    ]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Events", total_logs)
        error_pct = (error_count / total_logs * 100) if total_logs > 0 else 0
        st.metric("Errors", error_count, delta=f"{error_pct:.1f}%")
    
    with col2:
        warning_pct = (warning_count / total_logs * 100) if total_logs > 0 else 0
        st.metric("Warnings", warning_count, delta=f"{warning_pct:.1f}%")
        info_pct = (info_count / total_logs * 100) if total_logs > 0 else 0
        st.metric("Info", info_count, delta=f"{info_pct:.1f}%")
    
    with col3:
        st.metric("Last Hour", len(recent_logs))
        
        # Most active source
        source_counts = {}
        for log in logs:
            source = log.get('source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
        
        if source_counts:
            top_source = max(source_counts.items(), key=lambda x: x[1])
            st.metric("Most Active", f"{top_source[0]} ({top_source[1]})")
    
    with col4:
        # Latest log time (using Saudi time)
        if logs:
            latest_log = max(logs, key=lambda x: x.get('timestamp', datetime.min))
            latest_time = latest_log.get('timestamp')
            
            if latest_time:
                try:
                    time_ago_str = time_ago_saudi(latest_time)
                    st.metric("Latest Event", time_ago_str)
                except:
                    st.metric("Latest Event", "Unknown")
            else:
                st.metric("Latest Event", "Unknown")
    
    # Level distribution chart
    if logs:
        level_counts = {}
        for log in logs:
            level = log.get('level', 'INFO')
            level_counts[level] = level_counts.get(level, 0) + 1
        
        if len(level_counts) > 1:
            import plotly.graph_objects as go
            
            # Color mapping for log levels
            colors = {
                'INFO': 'green',
                'WARNING': 'orange',
                'ERROR': 'red',
                'DEBUG': 'blue'
            }
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(level_counts.keys()),
                    y=list(level_counts.values()),
                    marker_color=[colors.get(level, 'gray') for level in level_counts.keys()]
                )
            ])
            
            fig.update_layout(
                title="Log Level Distribution",
                xaxis_title="Log Level",
                yaxis_title="Count",
                height=300,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)

def render_log_entries(logs: List[Dict[str, Any]]) -> None:
    """
    Render individual log entries
    
    Args:
        logs: List of log entries to display
    """
    
    if not logs:
        st.info("No log entries match the current filters")
        return
    
    st.subheader(f"📋 Event Log ({len(logs)} entries)")
    
    # Pagination
    page_size = 50
    total_pages = (len(logs) + page_size - 1) // page_size
    
    if total_pages > 1:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            page = st.selectbox(
                f"Page (showing {min(page_size, len(logs))} of {len(logs)} events)",
                options=list(range(1, total_pages + 1)),
                index=0
            )
        
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, len(logs))
        page_logs = logs[start_idx:end_idx]
    else:
        page_logs = logs
    
    # Display logs with styling
    for i, log in enumerate(page_logs):
        render_single_log_entry(log, i)

def render_single_log_entry(log: Dict[str, Any], index: int) -> None:
    """
    Render a single log entry with styling
    
    Args:
        log: Log entry dictionary
        index: Entry index for unique keys
    """
    
    level = log.get('level', 'INFO')
    message = log.get('message', 'No message')
    source = log.get('source', 'unknown')
    timestamp = log.get('timestamp')
    details = log.get('details', {})
    
    # Format timestamp in Saudi local time
    if timestamp:
        try:
            time_str = format_saudi_time_full(timestamp)
        except:
            time_str = str(timestamp) if timestamp else "Unknown time"
    else:
        time_str = "Unknown time"
    
    # Determine styling based on log level
    level_config = {
        'ERROR': {'color': '#ffebee', 'icon': '🔴', 'text_color': '#c62828'},
        'WARNING': {'color': '#fff3e0', 'icon': '🟡', 'text_color': '#ef6c00'},
        'INFO': {'color': '#e8f5e8', 'icon': '🟢', 'text_color': '#2e7d32'},
        'DEBUG': {'color': '#e3f2fd', 'icon': '🔵', 'text_color': '#1565c0'}
    }
    
    config = level_config.get(level, level_config['INFO'])
    
    # Create container with background color
    with st.container():
        col1, col2 = st.columns([1, 4])
        
        with col1:
            st.markdown(f"**{config['icon']} {level}**")
            st.caption(time_str)
            st.caption(f"📍 {source}")
        
        with col2:
            st.markdown(f"**{message}**")
            
            # Show details if available
            if details and isinstance(details, dict) and details:
                with st.expander("View Details", expanded=False):
                    # Format details nicely
                    if 'event_type' in details:
                        st.markdown(f"**Event Type:** {details['event_type']}")
                    
                    # Show other details
                    filtered_details = {k: v for k, v in details.items() if k != 'event_type'}
                    if filtered_details:
                        st.json(filtered_details)
        
        st.markdown("---")

def render_log_export(logs: List[Dict[str, Any]]) -> None:
    """
    Render log export options
    
    Args:
        logs: List of log entries to export
    """
    
    st.subheader("📤 Export Logs")
    
    if not logs:
        st.info("No logs to export")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export as JSON", use_container_width=True):
            # Convert timestamps to strings for JSON serialization
            export_logs = []
            for log in logs:
                export_log = log.copy()
                if isinstance(export_log.get('timestamp'), datetime):
                    export_log['timestamp'] = export_log['timestamp'].isoformat()
                export_logs.append(export_log)
            
            json_str = json.dumps(export_logs, indent=2, default=str)
            st.download_button(
                label="💾 Download JSON",
                data=json_str,
                file_name=f"forex_signals_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("📊 Export as CSV", use_container_width=True):
            # Convert to DataFrame
            df_data = []
            for log in logs:
                timestamp = log.get('timestamp')
                if isinstance(timestamp, datetime):
                    timestamp_str = timestamp.isoformat()
                else:
                    timestamp_str = str(timestamp) if timestamp else ''
                
                df_data.append({
                    'timestamp': timestamp_str,
                    'level': log.get('level', ''),
                    'source': log.get('source', ''),
                    'message': log.get('message', ''),
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
    
    with col3:
        if st.button("📋 Copy to Clipboard", use_container_width=True):
            # Format as readable text
            text_output = []
            for log in logs:
                timestamp = log.get('timestamp')
                if isinstance(timestamp, datetime):
                    time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    time_str = str(timestamp) if timestamp else 'Unknown'
                
                text_output.append(
                    f"[{time_str}] {log.get('level', 'INFO')} "
                    f"{log.get('source', 'unknown')}: {log.get('message', '')}"
                )
            
            st.code('\n'.join(text_output))
            st.success("Log text generated above - copy manually")

def render_log_search(logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Render advanced log search interface
    
    Args:
        logs: List of log entries
    
    Returns:
        Filtered list based on search criteria
    """
    
    with st.expander("🔍 Advanced Search"):
        col1, col2 = st.columns(2)
        
        with col1:
            message_search = st.text_input(
                "Message Contains",
                placeholder="Enter text to search in messages"
            )
            
            source_search = st.text_input(
                "Source Contains",
                placeholder="Enter source name or pattern"
            )
        
        with col2:
            level_multi = st.multiselect(
                "Log Levels",
                options=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                default=['INFO', 'WARNING', 'ERROR']
            )
            
            date_range = st.date_input(
                "Date Range",
                value=[datetime.now().date() - timedelta(days=1), datetime.now().date()],
                max_value=datetime.now().date()
            )
        
        # Apply advanced filters
        filtered_logs = logs.copy()
        
        if message_search:
            filtered_logs = [
                log for log in filtered_logs
                if message_search.lower() in log.get('message', '').lower()
            ]
        
        if source_search:
            filtered_logs = [
                log for log in filtered_logs
                if source_search.lower() in log.get('source', '').lower()
            ]
        
        if level_multi:
            filtered_logs = [
                log for log in filtered_logs
                if log.get('level') in level_multi
            ]
        
        if date_range and len(date_range) == 2:
            start_date, end_date = date_range
            filtered_logs = [
                log for log in filtered_logs
                if log.get('timestamp') and 
                start_date <= (
                    log['timestamp'].date() if isinstance(log['timestamp'], datetime)
                    else datetime.fromisoformat(log['timestamp'].replace('Z', '+00:00')).date()
                ) <= end_date
            ]
        
        return filtered_logs

def render_realtime_log_viewer(
    log_source_func,
    refresh_interval: int = 30,
    max_entries: int = 100
) -> None:
    """
    Render real-time log viewer with auto-refresh
    
    Args:
        log_source_func: Function that returns list of log entries
        refresh_interval: Refresh interval in seconds
        max_entries: Maximum number of entries to display
    """
    
    # Auto-refresh placeholder
    placeholder = st.empty()
    
    while True:
        with placeholder.container():
            try:
                logs = log_source_func()
                render_logs_viewer(
                    logs[-max_entries:] if logs else [],
                    show_filters=True,
                    show_export=False,
                    auto_refresh=True
                )
            except Exception as e:
                st.error(f"Error loading logs: {e}")
        
        # Sleep for refresh interval
        import time
        time.sleep(refresh_interval)
