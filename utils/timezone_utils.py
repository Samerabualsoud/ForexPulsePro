"""
Timezone Utilities for converting timestamps to Saudi Arabia local time (UTC+3)
"""
from datetime import datetime, timezone, timedelta
from typing import Union, Optional

# Saudi Arabia timezone (UTC+3)
SAUDI_TIMEZONE = timezone(timedelta(hours=3))

def to_saudi_time(dt: Union[str, datetime]) -> datetime:
    """
    Convert UTC datetime/string to Saudi Arabia local time (UTC+3)
    
    Args:
        dt: UTC datetime object or ISO string
    
    Returns:
        datetime object in Saudi Arabia timezone
    """
    if isinstance(dt, str):
        try:
            # Parse ISO string, handle 'Z' suffix
            if 'T' in dt:
                parsed_dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
            else:
                # Handle simple date format
                parsed_dt = datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
                parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            # If parsing fails, return current Saudi time
            return datetime.now(SAUDI_TIMEZONE)
    else:
        parsed_dt = dt
    
    # Ensure datetime is UTC aware
    if parsed_dt.tzinfo is None:
        parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)
    elif parsed_dt.tzinfo != timezone.utc:
        # Convert to UTC first if it's in another timezone
        parsed_dt = parsed_dt.astimezone(timezone.utc)
    
    # Convert to Saudi time
    return parsed_dt.astimezone(SAUDI_TIMEZONE)

def format_saudi_time(dt: Union[str, datetime], format_str: str = "%m/%d %H:%M") -> str:
    """
    Format datetime in Saudi Arabia local time
    
    Args:
        dt: UTC datetime object or ISO string
        format_str: Strftime format string
    
    Returns:
        Formatted time string in Saudi Arabia timezone
    """
    try:
        saudi_dt = to_saudi_time(dt)
        return saudi_dt.strftime(format_str)
    except:
        return "N/A"

def format_saudi_time_full(dt: Union[str, datetime]) -> str:
    """
    Format full datetime with Saudi Arabia timezone indicator
    
    Args:
        dt: UTC datetime object or ISO string
    
    Returns:
        Full formatted time string with timezone (e.g., "2025-09-14 19:30:00 AST")
    """
    try:
        saudi_dt = to_saudi_time(dt)
        return saudi_dt.strftime("%Y-%m-%d %H:%M:%S AST")
    except:
        return "Unknown time"

def get_saudi_now() -> datetime:
    """
    Get current time in Saudi Arabia timezone
    
    Returns:
        Current datetime in Saudi Arabia timezone
    """
    return datetime.now(SAUDI_TIMEZONE)

def time_ago_saudi(dt: Union[str, datetime]) -> str:
    """
    Calculate time ago relative to Saudi Arabia current time
    
    Args:
        dt: UTC datetime object or ISO string
    
    Returns:
        Human-readable time ago string (e.g., "2 hours ago", "just now")
    """
    try:
        saudi_dt = to_saudi_time(dt)
        now = get_saudi_now()
        
        time_diff = now - saudi_dt
        total_seconds = time_diff.total_seconds()
        
        if total_seconds < 60:
            return "Just now"
        elif total_seconds < 3600:
            minutes = int(total_seconds / 60)
            return f"{minutes}m ago"
        elif total_seconds < 86400:
            hours = int(total_seconds / 3600)
            return f"{hours}h ago"
        else:
            days = int(total_seconds / 86400)
            return f"{days}d ago"
    except:
        return "Unknown"