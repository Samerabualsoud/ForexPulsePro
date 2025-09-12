"""
Base Data Provider Interface
"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional

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
