"""
Base Data Provider Interface with Real-Time Validation Support
"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import time

class BaseDataProvider(ABC):
    """Abstract base class for data providers with real-time validation support"""
    
    def __init__(self):
        self.name = getattr(self, 'name', 'Unknown Provider')
        self.is_live_source = getattr(self, 'is_live_source', False)
    
    @abstractmethod
    async def get_ohlc_data(
        self, 
        symbol: str, 
        timeframe: str = "M1", 
        limit: int = 100
    ) -> Optional[pd.DataFrame]:
        """
        Get OHLC data for a symbol with timestamp metadata
        
        Args:
            symbol: Currency pair (e.g., 'EURUSD')
            timeframe: Time interval (e.g., 'M1', 'M5', 'H1')
            limit: Number of bars to retrieve
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
            DataFrame.attrs must include: fetch_timestamp, data_source, is_real_data, is_live_source
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the data provider is available and configured"""
        pass
    
    @abstractmethod
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the latest price for a symbol"""
        pass
    
    @abstractmethod
    async def get_news(self, category: str = 'general', limit: int = 20) -> Optional[List[Dict[str, Any]]]:
        """
        Get financial news articles
        
        Args:
            category: News category ('general', 'forex', 'crypto', etc.)
            limit: Number of articles to retrieve
            
        Returns:
            List of news articles with metadata
        """
        pass
    
    @abstractmethod 
    async def get_symbol_news(self, symbol: str, limit: int = 10) -> Optional[List[Dict[str, Any]]]:
        """
        Get news articles related to a specific symbol/ticker
        
        Args:
            symbol: Symbol to get news for (e.g., 'EURUSD', 'BTCUSD')
            limit: Number of articles to retrieve
            
        Returns:
            List of news articles related to the symbol
        """
        pass
    
    def _add_metadata_to_dataframe(self, df: pd.DataFrame, symbol: str, data_source: str = None, last_updated: str = None) -> pd.DataFrame:
        """Add real-time validation metadata to DataFrame"""
        if df is None or df.empty:
            return df
            
        # Record exact fetch timestamp for freshness validation
        fetch_time = time.time()  # Unix timestamp with high precision
        fetch_iso = datetime.now(timezone.utc).isoformat()
        
        # Set data attributes for validation
        df.attrs = {
            'fetch_timestamp': fetch_time,  # Primary timestamp for freshness check
            'fetch_time_iso': fetch_iso,    # Human-readable timestamp
            'last_updated': last_updated or fetch_iso,  # Provider's data timestamp
            'data_source': data_source or self.name,
            'is_real_data': True,  # Mark as real market data
            'is_live_source': self.is_live_source,  # Whether provider offers live data
            'symbol': symbol,
            'validation_version': '2.0'  # Version for tracking validation logic
        }
        
        return df
    
    def _log_data_fetch(self, symbol: str, success: bool, data_size: int = 0, error: str = None):
        """Log data fetch attempts for debugging"""
        import structlog
        logger = structlog.get_logger(__name__)
        
        if success:
            logger.debug(f"{self.name}: Successfully fetched {data_size} bars for {symbol}")
        else:
            logger.warning(f"{self.name}: Failed to fetch data for {symbol}: {error}")
    
    def _validate_price_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """Basic validation for price data integrity"""
        if df is None or df.empty:
            return False
            
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            return False
            
        # Check for invalid price values
        for col in required_columns:
            if (df[col] <= 0).any() or df[col].isna().any():
                return False
                
        return True
