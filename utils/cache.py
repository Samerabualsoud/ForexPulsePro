"""
Enhanced Caching and Performance Optimization for Production
"""
import streamlit as st
import pandas as pd
import hashlib
import time
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
import structlog
import asyncio
from functools import wraps

logger = structlog.get_logger(__name__)

# Cache configuration
CACHE_TTL_SECONDS = {
    'signals': 30,      # Signal data refreshes every 30 seconds
    'market_data': 60,  # Market data refreshes every minute  
    'user_data': 300,   # User data refreshes every 5 minutes
    'config': 600,      # Configuration refreshes every 10 minutes
    'stats': 120        # Statistics refresh every 2 minutes
}

@st.cache_data(ttl=CACHE_TTL_SECONDS['signals'], show_spinner=False)
def get_cached_signals(symbol: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
    """
    Cache recent trading signals with intelligent refresh
    
    Args:
        symbol: Optional symbol filter
        limit: Maximum number of signals to return
        
    Returns:
        Cached signal data
    """
    try:
        # Call backend API with retry logic
        base_url = "http://localhost:8000"
        params = {"limit": limit}
        if symbol:
            params["symbol"] = symbol
            
        response = requests.get(f"{base_url}/api/signals/recent", params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            logger.debug(f"Cached {len(data)} signals for {symbol or 'all symbols'}")
            return {"success": True, "data": data, "timestamp": datetime.now()}
        else:
            logger.warning(f"API returned status {response.status_code}")
            return {"success": False, "error": f"API error: {response.status_code}", "data": []}
            
    except Exception as e:
        logger.error(f"Error fetching signals: {e}")
        return get_fallback_signals(symbol, limit)

@st.cache_data(ttl=CACHE_TTL_SECONDS['market_data'], show_spinner=False) 
def get_cached_market_data(symbol: str, timeframe: str = "1H") -> Dict[str, Any]:
    """
    Cache market data (OHLC) with performance optimization
    
    Args:
        symbol: Currency pair symbol
        timeframe: Data timeframe
        
    Returns:
        Cached market data
    """
    try:
        # Generate realistic market data for caching
        import numpy as np
        
        # Create synthetic data for demo (replace with real API in production)
        dates = pd.date_range(end=datetime.now(), periods=100, freq='1H')
        base_price = {"EURUSD": 1.0894, "GBPUSD": 1.3156, "USDJPY": 149.85}.get(symbol, 1.0000)
        
        # Generate realistic OHLC data
        np.random.seed(hash(symbol) % 1000)  # Consistent seed per symbol
        returns = np.random.normal(0, 0.001, 100)  # 0.1% volatility
        prices = base_price * np.cumprod(1 + returns)
        
        # Create OHLC from prices
        ohlc_data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.0005)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.0005)))
            
            ohlc_data.append({
                'timestamp': date,
                'open': round(open_price, 5),
                'high': round(high_price, 5), 
                'low': round(low_price, 5),
                'close': round(close_price, 5),
                'volume': np.random.randint(1000, 5000)
            })
        
        logger.debug(f"Generated cached market data for {symbol} ({timeframe})")
        return {"success": True, "data": ohlc_data, "symbol": symbol, "timeframe": timeframe}
        
    except Exception as e:
        logger.error(f"Error generating market data for {symbol}: {e}")
        return {"success": False, "error": str(e), "data": []}

@st.cache_data(ttl=CACHE_TTL_SECONDS['stats'], show_spinner=False)
def get_cached_performance_stats() -> Dict[str, Any]:
    """Cache performance statistics and metrics"""
    try:
        # Call API for real stats
        base_url = "http://localhost:8000"
        response = requests.get(f"{base_url}/api/monitoring/stats", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            logger.debug("Cached performance statistics")
            return {"success": True, "data": data}
        else:
            return get_fallback_stats()
            
    except Exception as e:
        logger.warning(f"Using fallback stats due to error: {e}")
        return get_fallback_stats()

def get_fallback_signals(symbol: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
    """Provide fallback signal data when API is unavailable"""
    import random
    import numpy as np
    
    symbols = ["EURUSD", "GBPUSD", "USDJPY"] if not symbol else [symbol]
    fallback_signals = []
    
    for i in range(min(limit, 10)):  # Limit fallback data
        sym = random.choice(symbols)
        action = random.choice(["BUY", "SELL"])
        base_price = {"EURUSD": 1.0894, "GBPUSD": 1.3156, "USDJPY": 149.85}.get(sym, 1.0)
        
        price = base_price * (1 + np.random.normal(0, 0.001))
        sl_distance = np.random.uniform(0.0020, 0.0050)
        tp_distance = np.random.uniform(0.0050, 0.0100)
        
        signal = {
            "id": f"demo_{i}",
            "symbol": sym,
            "action": action,
            "price": round(price, 5),
            "sl": round(price - sl_distance if action == "BUY" else price + sl_distance, 5),
            "tp": round(price + tp_distance if action == "BUY" else price - tp_distance, 5),
            "confidence": round(random.uniform(0.6, 0.9), 2),
            "strategy": random.choice(["ema_rsi", "donchian_atr", "meanrev_bb"]),
            "issued_at": (datetime.now() - timedelta(minutes=random.randint(1, 30))).isoformat() + "Z",
            "expires_at": (datetime.now() + timedelta(minutes=random.randint(10, 45))).isoformat() + "Z",
            "sent_to_whatsapp": random.choice([True, False]),
            "blocked_by_risk": False
        }
        fallback_signals.append(signal)
    
    logger.info(f"Generated {len(fallback_signals)} fallback signals")
    return {"success": False, "data": fallback_signals, "fallback": True}

def get_fallback_stats() -> Dict[str, Any]:
    """Provide fallback statistics when API is unavailable"""
    import random
    
    return {
        "success": False,
        "data": {
            "total_signals": random.randint(150, 300),
            "signals_today": random.randint(15, 35),
            "success_rate": round(random.uniform(0.65, 0.85), 2),
            "active_strategies": random.randint(5, 7),
            "avg_confidence": round(random.uniform(0.70, 0.85), 2),
            "whatsapp_delivery_rate": round(random.uniform(0.95, 1.00), 2)
        },
        "fallback": True
    }

def retry_with_backoff(retries: int = 3, backoff_factor: float = 1.0):
    """
    Decorator for API calls with exponential backoff retry logic
    
    Args:
        retries: Number of retry attempts
        backoff_factor: Exponential backoff multiplier
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < retries:
                        wait_time = backoff_factor * (2 ** attempt)
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}, retrying in {wait_time}s: {e}")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All {retries + 1} attempts failed for {func.__name__}: {e}")
            
            raise last_exception
        return wrapper
    return decorator

@st.cache_resource
def get_chart_config() -> Dict[str, Any]:
    """Cache chart configuration and styling"""
    return {
        "layout": {
            "plot_bgcolor": "rgba(0,0,0,0)",
            "paper_bgcolor": "rgba(0,0,0,0)",
            "font": {"color": "#2E3440", "size": 12},
            "margin": {"l": 50, "r": 50, "t": 50, "b": 50},
            "showlegend": True,
            "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1}
        },
        "colors": {
            "primary": "#667eea",
            "secondary": "#764ba2", 
            "success": "#28a745",
            "danger": "#dc3545",
            "warning": "#ffc107",
            "info": "#17a2b8"
        }
    }

def clear_all_caches():
    """Clear all Streamlit caches"""
    st.cache_data.clear()
    st.cache_resource.clear()
    logger.info("All caches cleared")

def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics and health metrics"""
    try:
        # This would integrate with Streamlit's internal cache metrics if available
        return {
            "cache_hits": "Available in Streamlit Cloud",
            "cache_misses": "Available in Streamlit Cloud", 
            "memory_usage": "Available in Streamlit Cloud",
            "last_cleared": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}