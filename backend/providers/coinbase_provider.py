"""
Coinbase Advanced API Provider for Real-Time Crypto OHLC Data
"""
import asyncio
import httpx
import pandas as pd
import structlog
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any
import time

from .base import BaseDataProvider

logger = structlog.get_logger(__name__)

class CoinbaseProvider(BaseDataProvider):
    """Coinbase Advanced API provider for real-time crypto market data"""
    
    def __init__(self):
        super().__init__()
        self.name = "Coinbase"
        self.is_live_source = True  # Coinbase provides real-time data
        self.base_url = "https://api.exchange.coinbase.com"
        
        # Crypto symbol mapping for Coinbase API
        self.crypto_mapping = {
            'BTCUSD': 'BTC-USD',
            'ETHUSD': 'ETH-USD', 
            'ADAUSD': 'ADA-USD',
            'DOGEUSD': 'DOGE-USD',
            'SOLUSD': 'SOL-USD',
            'BNBUSD': None,  # BNB not available on Coinbase
            'XRPUSD': 'XRP-USD',
            'MATICUSD': 'MATIC-USD',
            # Additional popular crypto pairs
            'LTCUSD': 'LTC-USD',
            'LINKUSD': 'LINK-USD',
            'DOTUSD': 'DOT-USD',
            'AVAXUSD': 'AVAX-USD',
            'UNIUSD': 'UNI-USD',
            'ATOMUSD': 'ATOM-USD',
        }
        
        # Granularity mapping (in seconds)
        self.granularity_mapping = {
            'M1': 60,      # 1 minute
            'M5': 300,     # 5 minutes
            'M15': 900,    # 15 minutes
            'M30': 1800,   # 30 minutes
            'H1': 3600,    # 1 hour
            'H4': 14400,   # 4 hours
            'D1': 86400,   # 1 day
            'W1': 604800,  # 1 week
        }
        
        logger.info(f"Coinbase provider initialized for real crypto market data")
        
    def is_available(self) -> bool:
        """Check if Coinbase API is available (no API key required for public data)"""
        return True
    
    def _get_coinbase_symbol(self, symbol: str) -> Optional[str]:
        """Convert standard crypto symbol to Coinbase format"""
        return self.crypto_mapping.get(symbol.upper())
    
    def _get_coinbase_granularity(self, timeframe: str) -> int:
        """Convert timeframe to Coinbase granularity (in seconds)"""
        return self.granularity_mapping.get(timeframe.upper(), 3600)  # Default 1 hour
    
    async def get_ohlc_data(self, symbol: str, timeframe: str = "H1", limit: int = 100) -> Optional[pd.DataFrame]:
        """Get real OHLC data from Coinbase"""
        if not self.is_available():
            return None
            
        try:
            coinbase_symbol = self._get_coinbase_symbol(symbol)
            if not coinbase_symbol:
                logger.warning(f"Coinbase: Symbol {symbol} not supported")
                return None
                
            granularity = self._get_coinbase_granularity(timeframe)
            
            # Calculate time range for historical data
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(seconds=granularity * limit)
            
            # Make request to Coinbase Candles API
            url = f"{self.base_url}/products/{coinbase_symbol}/candles"
            params = {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'granularity': granularity
            }
            
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
            if not data:
                logger.warning(f"Coinbase: No data returned for {symbol}")
                return None
                
            # Convert Coinbase data to DataFrame
            # Coinbase candles format: [timestamp, low, high, open, close, volume]
            df_data = []
            for candle in reversed(data):  # Coinbase returns newest first, we want oldest first
                try:
                    df_data.append({
                        'timestamp': pd.to_datetime(int(candle[0]), unit='s'),
                        'open': float(candle[3]),
                        'high': float(candle[2]),
                        'low': float(candle[1]),
                        'close': float(candle[4]),
                        'volume': float(candle[5])
                    })
                except (ValueError, IndexError) as e:
                    logger.warning(f"Coinbase: Invalid candle data for {symbol}: {e}")
                    continue
                    
            if not df_data:
                logger.warning(f"Coinbase: No valid candle data for {symbol}")
                return None
                
            df = pd.DataFrame(df_data)
            
            # Validate the data
            if not self._validate_price_data(df, symbol):
                logger.error(f"Coinbase: Invalid price data for {symbol}")
                return None
                
            # Add metadata for real-time validation
            df = self._add_metadata_to_dataframe(df, symbol, data_source="Coinbase")
            
            logger.info(f"Coinbase: Successfully fetched {len(df)} OHLC bars for {symbol}")
            return df
            
        except httpx.HTTPError as e:
            logger.error(f"Coinbase API request failed for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to get OHLC data for {symbol} from Coinbase: {e}")
            return None
    
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the latest price for a symbol from Coinbase"""
        try:
            coinbase_symbol = self._get_coinbase_symbol(symbol)
            if not coinbase_symbol:
                return None
                
            url = f"{self.base_url}/products/{coinbase_symbol}/ticker"
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
                
            return float(data['price'])
            
        except Exception as e:
            logger.error(f"Failed to get latest price for {symbol} from Coinbase: {e}")
            return None
    
    async def get_news(self, category: str = 'general', limit: int = 20) -> Optional[List[Dict[str, Any]]]:
        """Coinbase doesn't provide news API - return None"""
        return None
    
    async def get_symbol_news(self, symbol: str, limit: int = 10) -> Optional[List[Dict[str, Any]]]:
        """Coinbase doesn't provide news API - return None"""  
        return None