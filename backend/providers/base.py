"""
Base Data Provider Interface with Real-Time Validation Support
Includes FX pair normalization to prevent inversion issues
"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import time
from ..instruments.metadata import forex_normalizer
import structlog

class BaseDataProvider(ABC):
    """Abstract base class for data providers with real-time validation support"""
    
    def __init__(self):
        self.name = getattr(self, 'name', 'Unknown Provider')
        self.is_live_source = getattr(self, 'is_live_source', False)
        self.logger = structlog.get_logger(f"{self.__class__.__name__}")
    
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
    
    def normalize_symbol(self, symbol: str) -> str:
        """
        Normalize a forex symbol to its standard canonical form.
        
        This ensures all providers use consistent symbol orientations,
        preventing AUDUSD vs USDAUD inversion issues.
        
        Args:
            symbol: Raw symbol from provider
            
        Returns:
            Normalized standard symbol
        """
        normalized = forex_normalizer.normalize_symbol(symbol)
        
        if normalized != symbol:
            self.logger.info(f"🔄 {self.name}: Normalized symbol {symbol} → {normalized}")
        
        return normalized
    
    def normalize_ohlc_data(self, df: pd.DataFrame, original_symbol: str) -> pd.DataFrame:
        """
        Apply FX pair normalization to OHLC data.
        
        If the provider returned inverted data (e.g., USDAUD instead of AUDUSD),
        this function inverts all OHLC prices to match the standard orientation.
        
        Args:
            df: OHLC DataFrame from provider
            original_symbol: Original symbol from provider
            
        Returns:
            DataFrame with normalized OHLC data and updated metadata
        """
        if df is None or df.empty:
            return df
            
        # Get normalized symbol
        normalized_symbol = self.normalize_symbol(original_symbol)
        
        # Apply OHLC normalization if needed
        normalized_df = forex_normalizer.normalize_ohlc_data(df, original_symbol, normalized_symbol)
        
        # Update DataFrame metadata with normalization info
        if hasattr(normalized_df, 'attrs'):
            normalized_df.attrs['original_symbol'] = original_symbol
            normalized_df.attrs['normalized_symbol'] = normalized_symbol
            normalized_df.attrs['pair_inverted'] = forex_normalizer.is_inverted(original_symbol, normalized_symbol)
            normalized_df.attrs['normalization_applied'] = True
            normalized_df.attrs['provider_name'] = self.name
        
        return normalized_df
    
    def normalize_price(self, price: float, original_symbol: str) -> float:
        """
        Apply FX pair normalization to a single price value.
        
        Args:
            price: Original price from provider
            original_symbol: Original symbol from provider
            
        Returns:
            Normalized price (inverted if necessary)
        """
        if price is None:
            return price
            
        normalized_symbol = self.normalize_symbol(original_symbol)
        normalized_price = forex_normalizer.normalize_price(price, original_symbol, normalized_symbol)
        
        return normalized_price
    
    def get_normalization_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get detailed normalization information for debugging.
        
        Args:
            symbol: Symbol to analyze
            
        Returns:
            Dictionary with normalization details
        """
        return forex_normalizer.get_normalization_info(symbol)
    
    def validate_normalized_data(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate that data has been properly normalized.
        
        Used for debugging and ensuring data consistency.
        
        Args:
            symbol: Expected normalized symbol
            df: DataFrame to validate
            
        Returns:
            Validation results
        """
        validation_results = {
            'symbol': symbol,
            'is_normalized': False,
            'has_metadata': False,
            'inversion_applied': False,
            'issues': []
        }
        
        # Check if DataFrame has normalization metadata
        if hasattr(df, 'attrs'):
            validation_results['has_metadata'] = True
            
            if 'normalized_symbol' in df.attrs:
                validation_results['is_normalized'] = True
                normalized_symbol = df.attrs['normalized_symbol']
                
                if normalized_symbol != symbol:
                    validation_results['issues'].append(f"Symbol mismatch: expected {symbol}, got {normalized_symbol}")
                    
                if df.attrs.get('pair_inverted', False):
                    validation_results['inversion_applied'] = True
                    
        else:
            validation_results['issues'].append("DataFrame missing normalization metadata")
        
        # Check for data integrity after normalization
        if not self._validate_price_data(df, symbol):
            validation_results['issues'].append("Price data validation failed after normalization")
            
        return validation_results
