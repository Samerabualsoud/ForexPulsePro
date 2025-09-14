"""
Binance API Provider for Real-Time Crypto OHLC Data
"""
import asyncio
import httpx
import pandas as pd
import structlog
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
import time

from .base import BaseDataProvider

logger = structlog.get_logger(__name__)

class BinanceProvider(BaseDataProvider):
    """Binance API provider for real-time crypto market data"""
    
    def __init__(self):
        super().__init__()
        self.name = "Binance"
        self.is_live_source = True  # Binance provides real-time data
        self.base_url = "https://api.binance.com"
        
        # Crypto symbol mapping for Binance API
        self.crypto_mapping = {
            'BTCUSD': 'BTCUSDT',
            'ETHUSD': 'ETHUSDT', 
            'ADAUSD': 'ADAUSDT',
            'DOGEUSD': 'DOGEUSDT',
            'SOLUSD': 'SOLUSDT',
            'BNBUSD': 'BNBUSDT',
            'XRPUSD': 'XRPUSDT',
            'MATICUSD': 'MATICUSDT',
            # Additional popular crypto pairs
            'LTCUSD': 'LTCUSDT',
            'LINKUSD': 'LINKUSDT',
            'DOTUSD': 'DOTUSDT',
            'AVAXUSD': 'AVAXUSDT',
            'UNIUSD': 'UNIUSDT',
            'ATOMUSD': 'ATOMUSDT',
        }
        
        # Timeframe mapping
        self.timeframe_mapping = {
            'M1': '1m',
            'M5': '5m', 
            'M15': '15m',
            'M30': '30m',
            'H1': '1h',
            'H4': '4h',
            'D1': '1d',
            'W1': '1w',
        }
        
        logger.info(f"Binance provider initialized for real crypto market data")
        
    def is_available(self) -> bool:
        """Check if Binance API is available (no API key required for public data)"""
        return True
    
    def _get_binance_symbol(self, symbol: str) -> Optional[str]:
        """Convert standard crypto symbol to Binance format"""
        return self.crypto_mapping.get(symbol.upper())
    
    def _get_binance_interval(self, timeframe: str) -> str:
        """Convert timeframe to Binance interval format"""
        return self.timeframe_mapping.get(timeframe.upper(), '1h')
    
    async def get_ohlc_data(self, symbol: str, timeframe: str = "H1", limit: int = 100) -> Optional[pd.DataFrame]:
        """Get real OHLC data from Binance"""
        if not self.is_available():
            return None
            
        try:
            binance_symbol = self._get_binance_symbol(symbol)
            if not binance_symbol:
                logger.warning(f"Binance: Symbol {symbol} not supported")
                return None
                
            interval = self._get_binance_interval(timeframe)
            
            # Make request to Binance Klines API
            url = f"{self.base_url}/api/v3/klines"
            params = {
                'symbol': binance_symbol,
                'interval': interval,
                'limit': min(limit, 1000)  # Binance limit is 1000
            }
            
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
            if not data:
                logger.warning(f"Binance: No data returned for {symbol}")
                return None
                
            # Convert Binance data to DataFrame
            # Binance klines format: [timestamp, open, high, low, close, volume, close_time, quote_asset_volume, number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore]
            df_data = []
            for kline in data:
                df_data.append({
                    'timestamp': pd.to_datetime(int(kline[0]), unit='ms'),
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5])
                })
                
            df = pd.DataFrame(df_data)
            
            # Validate the data
            if not self._validate_price_data(df, symbol):
                logger.error(f"Binance: Invalid price data for {symbol}")
                return None
                
            # Add metadata for real-time validation
            df = self._add_metadata_to_dataframe(df, symbol, data_source="Binance")
            
            logger.info(f"Binance: Successfully fetched {len(df)} OHLC bars for {symbol}")
            return df
            
        except httpx.HTTPError as e:
            logger.error(f"Binance API request failed for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to get OHLC data for {symbol} from Binance: {e}")
            return None
    
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the latest price for a symbol from Binance"""
        try:
            binance_symbol = self._get_binance_symbol(symbol)
            if not binance_symbol:
                return None
                
            url = f"{self.base_url}/api/v3/ticker/price"
            params = {'symbol': binance_symbol}
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
            return float(data['price'])
            
        except Exception as e:
            logger.error(f"Failed to get latest price for {symbol} from Binance: {e}")
            return None
    
    async def get_news(self, category: str = 'general', limit: int = 20) -> Optional[List[Dict[str, Any]]]:
        """Binance doesn't provide news API - return None"""
        return None
    
    async def get_symbol_news(self, symbol: str, limit: int = 10) -> Optional[List[Dict[str, Any]]]:
        """Binance doesn't provide news API - return None"""  
        return None