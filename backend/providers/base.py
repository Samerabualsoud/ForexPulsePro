"""
Base Data Provider Interface
"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, List, Dict, Any

class BaseDataProvider(ABC):
    """Abstract base class for data providers"""
    
    @abstractmethod
    async def get_ohlc_data(
        self, 
        symbol: str, 
        timeframe: str = "M1", 
        limit: int = 100
    ) -> Optional[pd.DataFrame]:
        """
        Get OHLC data for a symbol
        
        Args:
            symbol: Currency pair (e.g., 'EURUSD')
            timeframe: Time interval (e.g., 'M1', 'M5', 'H1')
            limit: Number of bars to retrieve
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
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
