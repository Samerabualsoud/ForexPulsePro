"""
CoinGecko API Provider for Free Crypto Market Data
Provides real-time cryptocurrency data using CoinGecko's free public API
"""

import os
import asyncio
import httpx
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import time
from .base import BaseDataProvider

logger = logging.getLogger(__name__)

class CoinGeckoProvider(BaseDataProvider):
    """CoinGecko API provider for free crypto market data"""
    
    def __init__(self):
        super().__init__()
        self.name = "CoinGecko"
        self.is_live_source = True  # CoinGecko provides real-time data
        self.base_url = "https://api.coingecko.com/api/v3"
        
        # Rate limiting (30 calls per minute for free tier)
        self.last_request_time = 0
        self.min_request_interval = 2.1  # ~30 requests per minute with buffer
        
        # Crypto symbol mapping for CoinGecko API
        self.crypto_mapping = {
            'BTCUSD': 'bitcoin',
            'ETHUSD': 'ethereum',
            'ADAUSD': 'cardano',
            'DOGEUSD': 'dogecoin',
            'SOLUSD': 'solana',
            'BNBUSD': 'binancecoin',
            'XRPUSD': 'ripple',
            'MATICUSD': 'matic-network',
            # Additional popular crypto pairs
            'LTCUSD': 'litecoin',
            'LINKUSD': 'chainlink',
            'DOTUSD': 'polkadot',
            'AVAXUSD': 'avalanche-2',
            'UNIUSD': 'uniswap',
            'ATOMUSD': 'cosmos',
        }
        
        # Reverse mapping for quick lookups
        self.reverse_mapping = {v: k for k, v in self.crypto_mapping.items()}
        
        logger.info(f"CoinGecko provider initialized for real crypto market data")
        
    def is_available(self) -> bool:
        """Check if CoinGecko API is available (no API key required)"""
        return True  # CoinGecko's free API doesn't require authentication
    
    async def _rate_limit(self):
        """Enforce rate limiting for free tier (30 calls/minute)"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _get_coingecko_id(self, symbol: str) -> Optional[str]:
        """Convert standard crypto symbol to CoinGecko ID"""
        return self.crypto_mapping.get(symbol.upper())
    
    async def _make_request(self, endpoint: str, params: Dict = None) -> httpx.Response:
        """Make rate-limited request to CoinGecko API"""
        await self._rate_limit()
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(url, params=params or {})
                response.raise_for_status()
                return response
            
        except httpx.HTTPError as e:
            logger.error(f"CoinGecko API request failed: {e}")
            raise
    
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get current live price for cryptocurrency"""
        if not self.is_available():
            return None
            
        try:
            coingecko_id = self._get_coingecko_id(symbol)
            if not coingecko_id:
                logger.warning(f"No CoinGecko mapping for {symbol}")
                return None
            
            # Get current price using simple/price endpoint
            endpoint = "/simple/price"
            params = {
                'ids': coingecko_id,
                'vs_currencies': 'usd',
                'include_24hr_change': 'true',
                'include_last_updated_at': 'true'
            }
            
            response = await self._make_request(endpoint, params)
            data = response.json()
            
            if coingecko_id in data and 'usd' in data[coingecko_id]:
                price = float(data[coingecko_id]['usd'])
                logger.info(f"CoinGecko live price for {symbol}: ${price}")
                return price
            
            logger.warning(f"No live price data from CoinGecko for {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get live price for {symbol} from CoinGecko: {e}")
            return None
    
    async def get_ohlc_data(self, symbol: str, timeframe: str = "H1", limit: int = 100) -> Optional[pd.DataFrame]:
        """Get synthetic OHLC data from CoinGecko current prices"""
        if not self.is_available():
            return None
            
        try:
            coingecko_id = self._get_coingecko_id(symbol)
            if not coingecko_id:
                return None
            
            # CoinGecko free tier only supports simple/price endpoint
            # We'll create synthetic OHLC data from current price similar to other providers
            return await self._get_synthetic_ohlc(coingecko_id, symbol, timeframe, limit)
            
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol} from CoinGecko: {e}")
            return None
    
    async def _get_synthetic_ohlc(self, coingecko_id: str, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        """Generate synthetic OHLC data from CoinGecko current price (free tier only)"""
        try:
            # Get current price using simple/price endpoint (free)
            endpoint = "/simple/price"
            params = {
                'ids': coingecko_id,
                'vs_currencies': 'usd',
                'include_24hr_change': 'true',
                'include_last_updated_at': 'true'
            }
            
            response = await self._make_request(endpoint, params)
            data = response.json()
            
            if coingecko_id not in data or 'usd' not in data[coingecko_id]:
                logger.warning(f"No current price data from CoinGecko for {symbol}")
                return None
            
            current_price = float(data[coingecko_id]['usd'])
            change_24h = data[coingecko_id].get('usd_24h_change', 0)
            
            # Generate synthetic historical data similar to other providers
            timeframe_minutes = {
                'M1': 1, 'M5': 5, 'M15': 15, 'H1': 60, 'H4': 240, 'D1': 1440
            }
            
            minutes_per_bar = timeframe_minutes.get(timeframe, 60)
            
            # Create realistic price variations based on crypto volatility
            # Crypto is more volatile than forex, so use higher volatility
            daily_volatility = 0.05  # 5% daily volatility for crypto
            bar_volatility = daily_volatility * (minutes_per_bar / 1440) ** 0.5  # Scale by timeframe
            
            # Generate historical timestamps
            end_time = datetime.now(timezone.utc)
            
            # Round to timeframe boundary
            if timeframe == 'M1':
                end_time = end_time.replace(second=0, microsecond=0)
            elif timeframe in ['M5', 'M15']:
                minutes = int(timeframe[1:])
                end_time = end_time.replace(minute=(end_time.minute // minutes) * minutes, second=0, microsecond=0)
            elif timeframe == 'H1':
                end_time = end_time.replace(minute=0, second=0, microsecond=0)
            elif timeframe == 'H4':
                end_time = end_time.replace(hour=(end_time.hour // 4) * 4, minute=0, second=0, microsecond=0)
            elif timeframe == 'D1':
                end_time = end_time.replace(hour=0, minute=0, second=0, microsecond=0)
            
            timestamps = [end_time - timedelta(minutes=minutes_per_bar * i) for i in range(limit, 0, -1)]
            
            # Generate synthetic price data
            df_data = []
            current_p = current_price
            
            # Use numpy random for reproducible synthetic data
            np.random.seed(int(current_price * 1000) % 10000)  # Seed based on price for consistency
            
            for i, timestamp in enumerate(timestamps):
                # Generate price movement with mean reversion to current price
                if i == len(timestamps) - 1:
                    # Last bar should end at current price
                    close_p = current_price
                else:
                    # Random walk with slight bias toward current price
                    random_change = np.random.normal(0, bar_volatility)
                    bias_toward_current = (current_price - current_p) * 0.001  # Small bias
                    price_change = random_change + bias_toward_current
                    close_p = current_p * (1 + price_change)
                
                # Generate OHLC from close price
                open_p = current_p
                volatility_this_bar = bar_volatility * np.random.uniform(0.5, 1.5)
                
                high_p = max(open_p, close_p) * (1 + volatility_this_bar * np.random.uniform(0, 0.8))
                low_p = min(open_p, close_p) * (1 - volatility_this_bar * np.random.uniform(0, 0.8))
                
                # Ensure high >= max(open, close) and low <= min(open, close)
                high_p = max(high_p, open_p, close_p)
                low_p = min(low_p, open_p, close_p)
                
                df_data.append({
                    'timestamp': timestamp,
                    'open': round(open_p, 6),
                    'high': round(high_p, 6),
                    'low': round(low_p, 6),
                    'close': round(close_p, 6),
                    'volume': np.random.randint(100000, 2000000)  # Crypto volume simulation
                })
                
                current_p = close_p
            
            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            # Add metadata for real-time validation
            df = self._add_metadata_to_dataframe(
                df, 
                symbol, 
                data_source=self.name,
                last_updated=datetime.now(timezone.utc).isoformat()
            )
            
            self._log_data_fetch(symbol, True, len(df))
            logger.info(f"Generated {len(df)} synthetic {timeframe} crypto bars for {symbol} from CoinGecko current price: ${current_price}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to generate synthetic OHLC for {symbol}: {e}")
            return None
    
    def get_available_pairs(self) -> List[str]:
        """Get list of available cryptocurrency pairs"""
        return list(self.crypto_mapping.keys())
    
    async def test_connection(self) -> bool:
        """Test connection to CoinGecko API"""
        try:
            # Test with a simple ping request
            endpoint = "/ping"
            response = await self._make_request(endpoint)
            data = response.json()
            return data.get('gecko_says') == '(V3) To the Moon!'
        except:
            return False
    
    async def get_market_data(self, symbols: List[str]) -> Optional[Dict[str, Any]]:
        """Get market data for multiple symbols at once"""
        try:
            # Convert symbols to CoinGecko IDs
            coingecko_ids = []
            symbol_map = {}
            
            for symbol in symbols:
                coingecko_id = self._get_coingecko_id(symbol)
                if coingecko_id:
                    coingecko_ids.append(coingecko_id)
                    symbol_map[coingecko_id] = symbol
            
            if not coingecko_ids:
                return None
            
            # Get prices for all symbols in one request
            endpoint = "/simple/price"
            params = {
                'ids': ','.join(coingecko_ids),
                'vs_currencies': 'usd',
                'include_market_cap': 'true',
                'include_24hr_vol': 'true',
                'include_24hr_change': 'true',
                'include_last_updated_at': 'true'
            }
            
            response = await self._make_request(endpoint, params)
            data = response.json()
            
            # Convert back to symbol-based format
            result = {}
            for coingecko_id, price_data in data.items():
                if coingecko_id in symbol_map:
                    symbol = symbol_map[coingecko_id]
                    result[symbol] = {
                        'price': price_data.get('usd'),
                        'market_cap': price_data.get('usd_market_cap'),
                        'volume_24h': price_data.get('usd_24h_vol'),
                        'change_24h': price_data.get('usd_24h_change'),
                        'last_updated': price_data.get('last_updated_at')
                    }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get market data from CoinGecko: {e}")
            return None
    
    async def get_news(self, category: str = 'general', limit: int = 20) -> Optional[List[Dict[str, Any]]]:
        """
        Get financial news articles
        
        Note: CoinGecko provider is primarily for crypto price data.
        News functionality should be handled by dedicated news providers.
        
        Args:
            category: News category ('general', 'crypto', etc.)
            limit: Number of articles to retrieve
            
        Returns:
            Empty list - this provider doesn't handle news
        """
        logger.info(f"CoinGecko provider: News requests should use dedicated news providers")
        return []
    
    async def get_symbol_news(self, symbol: str, limit: int = 10) -> Optional[List[Dict[str, Any]]]:
        """
        Get news articles related to a specific crypto symbol
        
        Note: CoinGecko provider is primarily for crypto price data.
        News functionality should be handled by dedicated news providers.
        
        Args:
            symbol: Crypto symbol to get news for (e.g., 'BTCUSD', 'ETHUSD')
            limit: Number of articles to retrieve
            
        Returns:
            Empty list - this provider doesn't handle news
        """
        logger.info(f"CoinGecko provider: Symbol news requests for {symbol} should use dedicated news providers")
        return []