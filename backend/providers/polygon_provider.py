"""
Polygon.io API Provider for Real Live Market Data
Provides real-time and historical market data for forex, stocks, and crypto including Bitcoin
"""

import os
import asyncio
import httpx
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, timezone
import pandas as pd
import time
import random
from .base import BaseDataProvider

logger = logging.getLogger(__name__)

class PolygonProvider(BaseDataProvider):
    """Polygon.io API provider for real live market data"""
    
    def __init__(self):
        super().__init__()
        self.name = "Polygon.io"
        self.is_live_source = True  # Polygon.io provides real-time data
        self.base_url = "https://api.polygon.io"
        
        # API key from environment (REQUIRED - no default for security)
        self.api_key = os.getenv('POLYGON_API_KEY')
        
        # Enhanced rate limiting with token bucket approach
        self._rate_limit_lock = None  # Lazy initialization to avoid event loop binding issues
        self.call_timestamps = []  # Track request timestamps
        self.calls_per_minute = 8  # Increased limit with better fallback
        
        # Data caching to reduce API calls
        self.price_cache = {}  # Cache for price data
        self.ohlc_cache = {}   # Cache for OHLC data
        self.cache_duration = timedelta(seconds=60)  # 1-minute cache
        
        # Symbol mapping for Polygon.io (Forex, Crypto, Commodities)
        self.symbol_mapping = {
            # Forex pairs
            'EURUSD': 'C:EURUSD',
            'GBPUSD': 'C:GBPUSD', 
            'USDJPY': 'C:USDJPY',
            'AUDUSD': 'C:AUDUSD',
            'USDCAD': 'C:USDCAD',
            'USDCHF': 'C:USDCHF',
            'NZDUSD': 'C:NZDUSD',
            'EURGBP': 'C:EURGBP',
            'EURJPY': 'C:EURJPY',
            'GBPJPY': 'C:GBPJPY',
            # Crypto pairs
            'BTCUSD': 'X:BTCUSD',
            'ETHUSD': 'X:ETHUSD',
            'ADAUSD': 'X:ADAUSD',
            'DOGEUSD': 'X:DOGEUSD',
            'SOLUSD': 'X:SOLUSD',
            'BNBUSD': 'X:BNBUSD',
            'XRPUSD': 'X:XRPUSD',
            'MATICUSD': 'X:MATICUSD',
            # Commodity pairs
            'XAUUSD': 'C:XAUUSD',  # Gold
            'XAGUSD': 'C:XAGUSD',  # Silver
            'USOIL': 'C:USOIL',    # WTI Crude Oil
        }
        
        # Asset type mapping for API endpoints
        self.asset_types = {
            'C:': 'forex',     # Currency (forex and commodities) 
            'X:': 'crypto'     # Crypto
        }
        
        logger.info(f"Polygon.io provider initialized for real live market data")

    @property 
    def rate_limit_lock(self) -> asyncio.Lock:
        """Get rate limiting lock, creating it lazily in the current event loop"""
        if self._rate_limit_lock is None:
            try:
                # This will create the lock in the current event loop
                self._rate_limit_lock = asyncio.Lock()
            except RuntimeError:
                # No event loop running, this should not happen in async context
                # but we'll handle it gracefully
                logger.warning("No event loop running when creating rate limit lock")
                raise
        return self._rate_limit_lock
        
    def is_available(self) -> bool:
        """Check if Polygon.io API is available"""
        return bool(self.api_key)
    
    async def _check_and_wait_rate_limit(self):
        """Advanced rate limiting with token bucket approach"""
        async with self.rate_limit_lock:
            now = time.time()
            
            # Remove timestamps older than 1 minute (token bucket refill)
            self.call_timestamps = [ts for ts in self.call_timestamps if now - ts < 60]
            
            # If we're at the limit, wait until we can make a call
            while len(self.call_timestamps) >= self.calls_per_minute:
                # Calculate wait time until oldest call expires
                if self.call_timestamps:
                    oldest_call = min(self.call_timestamps)
                    wait_time = max(3, 60 - (now - oldest_call) + 2)  # Extra 2s buffer
                else:
                    wait_time = 15
                
                logger.info(f"Polygon.io rate limit reached, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                
                # Refresh timestamps after waiting
                now = time.time()
                self.call_timestamps = [ts for ts in self.call_timestamps if now - ts < 60]
            
            # Record this call
            self.call_timestamps.append(now)
    
    def _get_polygon_symbol(self, pair: str) -> Optional[str]:
        """Convert standard pair to Polygon.io format"""
        return self.symbol_mapping.get(pair)
    
    async def _make_request_with_retry(self, endpoint: str, params: Dict = None, max_retries: int = 3) -> Optional[httpx.Response]:
        """Make rate-limited request with exponential backoff retry"""
        url = f"{self.base_url}{endpoint}"
        
        # Add API key to params
        if params is None:
            params = {}
        params['apiKey'] = self.api_key
        
        for attempt in range(max_retries + 1):
            try:
                await self._check_and_wait_rate_limit()
                
                async with httpx.AsyncClient(timeout=25.0) as client:
                    response = await client.get(url, params=params)
                    
                    if response.status_code == 429:
                        # Rate limited - implement exponential backoff
                        if attempt < max_retries:
                            backoff_time = (2 ** attempt) * 3 + random.uniform(1, 2)  # 4-5s, 7-8s, 13-14s
                            logger.warning(f"Polygon.io 429 rate limit hit, attempt {attempt + 1}/{max_retries + 1}, waiting {backoff_time:.1f}s")
                            await asyncio.sleep(backoff_time)
                            continue
                        else:
                            logger.warning(f"Polygon.io rate limit exceeded after {max_retries + 1} attempts, fallback required")
                            return None
                    
                    response.raise_for_status()
                    return response
                    
            except httpx.TimeoutException:
                if attempt < max_retries:
                    wait_time = (2 ** attempt) * 2 + random.uniform(1, 3)
                    logger.warning(f"Polygon.io timeout, retrying in {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Polygon.io timeout after {max_retries + 1} attempts")
                    return None
                    
            except httpx.HTTPError as e:
                # Sanitize error message to prevent API key leakage
                error_msg = str(e).replace(self.api_key, '[REDACTED]') if self.api_key else str(e)
                logger.error(f"Polygon.io API request failed: {error_msg}")
                if attempt < max_retries:
                    await asyncio.sleep((2 ** attempt) + random.uniform(0.5, 1.5))
                    continue
                return None
        
        return None
    
    def _get_cache_key(self, symbol: str, data_type: str) -> str:
        """Generate cache key for data"""
        return f"{symbol}_{data_type}"
    
    def _is_cache_valid(self, cache_key: str, cache_dict: dict) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in cache_dict:
            return False
        
        cached_data = cache_dict[cache_key]
        if 'timestamp' not in cached_data:
            return False
            
        cache_time = cached_data['timestamp']
        return datetime.now() - cache_time < self.cache_duration
    
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get current live price for currency pair with caching"""
        if not self.is_available():
            return None
        
        # Check cache first
        cache_key = self._get_cache_key(symbol, "price")
        if self._is_cache_valid(cache_key, self.price_cache):
            logger.debug(f"Using cached price for {symbol}")
            return self.price_cache[cache_key]['price']
            
        try:
            polygon_symbol = self._get_polygon_symbol(symbol)
            if not polygon_symbol:
                logger.warning(f"No Polygon.io mapping for {symbol}")
                return None
            
            # Get latest quote using correct endpoint for forex/crypto
            if polygon_symbol.startswith('C:'):
                # Forex - use real-time currency API
                endpoint = f"/v1/last/currencies/{polygon_symbol}"
            else:
                # Crypto - use crypto endpoint
                endpoint = f"/v1/last/crypto/{polygon_symbol}"
            
            response = await self._make_request_with_retry(endpoint)
            if not response:
                logger.error(f"Failed to get price for {symbol} from Polygon.io after retries")
                # Return cached data if available, even if expired
                if cache_key in self.price_cache:
                    logger.info(f"Using expired cached price for {symbol} due to API failure")
                    return self.price_cache[cache_key]['price']
                return None
            
            data = response.json()
            
            if data.get('status') == 'OK' and 'results' in data:
                results = data['results']
                # Calculate mid price from bid/ask
                bid = results.get('bid', 0)
                ask = results.get('ask', 0)
                
                if bid > 0 and ask > 0:
                    mid_price = (bid + ask) / 2
                    
                    # Cache the result
                    self.price_cache[cache_key] = {
                        'price': mid_price,
                        'timestamp': datetime.now()
                    }
                    
                    logger.info(f"Polygon.io live price for {symbol}: {mid_price}")
                    return mid_price
            
            logger.warning(f"No live price data from Polygon.io for {symbol}")
            return None
            
        except Exception as e:
            # Sanitize error message to prevent API key leakage
            error_msg = str(e).replace(self.api_key, '[REDACTED]') if self.api_key else str(e)
            logger.error(f"Failed to get live price for {symbol} from Polygon.io: {error_msg}")
            # Return cached data if available
            if cache_key in self.price_cache:
                logger.info(f"Using cached price for {symbol} due to error")
                return self.price_cache[cache_key]['price']
            return None
    
    async def get_ohlc_data(self, symbol: str, timeframe: str = "H1", limit: int = 100) -> Optional[pd.DataFrame]:
        """Get historical OHLC data from Polygon.io with caching"""
        if not self.is_available():
            return None
        
        # Check cache first
        cache_key = self._get_cache_key(f"{symbol}_{timeframe}_{limit}", "ohlc")
        if self._is_cache_valid(cache_key, self.ohlc_cache):
            logger.debug(f"Using cached OHLC data for {symbol}")
            return self.ohlc_cache[cache_key]['data']
            
        try:
            polygon_symbol = self._get_polygon_symbol(symbol)
            if not polygon_symbol:
                return None
            
            # Convert timeframe to Polygon.io format
            timeframe_mapping = {
                'M1': (1, 'minute'),
                'M5': (5, 'minute'),
                'M15': (15, 'minute'),
                'H1': (1, 'hour'),
                'H4': (4, 'hour'),
                'D1': (1, 'day')
            }
            
            multiplier, timespan = timeframe_mapping.get(timeframe, (1, 'hour'))
            
            # Calculate date range for historical data
            end_date = datetime.now()
            
            # Adjust period based on timeframe
            if timespan == 'minute':
                start_date = end_date - timedelta(hours=limit)
            elif timespan == 'hour':
                start_date = end_date - timedelta(days=limit // 24 + 1)
            else:  # day
                start_date = end_date - timedelta(days=limit)
            
            # Format dates for Polygon.io API
            from_date = start_date.strftime('%Y-%m-%d')
            to_date = end_date.strftime('%Y-%m-%d')
            
            # Get aggregated bars
            endpoint = f"/v2/aggs/ticker/{polygon_symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
            
            params = {
                'adjusted': 'true',
                'sort': 'asc',
                'limit': limit
            }
            
            response = await self._make_request_with_retry(endpoint, params)
            if not response:
                logger.error(f"Failed to get OHLC data for {symbol} from Polygon.io after retries")
                # Return cached data if available, even if expired
                if cache_key in self.ohlc_cache:
                    logger.info(f"Using expired cached OHLC data for {symbol} due to API failure")
                    return self.ohlc_cache[cache_key]['data']
                return None
            
            data = response.json()
            
            if data.get('status') == 'OK' and 'results' in data and data['results']:
                # Convert to DataFrame
                results = data['results']
                df_data = []
                
                for bar in results:
                    df_data.append({
                        'timestamp': pd.to_datetime(bar['t'], unit='ms'),
                        'open': float(bar['o']),
                        'high': float(bar['h']),
                        'low': float(bar['l']),
                        'close': float(bar['c']),
                        'volume': float(bar.get('v', 0))
                    })
                
                df = pd.DataFrame(df_data)
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                
                # Limit to requested number of bars
                df = df.tail(limit)
                
                # Add metadata for real-time validation
                df = self._add_metadata_to_dataframe(
                    df, 
                    symbol, 
                    data_source=self.name,
                    last_updated=datetime.now(timezone.utc).isoformat()
                )
                
                # Cache the result
                self.ohlc_cache[cache_key] = {
                    'data': df,
                    'timestamp': datetime.now()
                }
                
                self._log_data_fetch(symbol, True, len(df))
                logger.info(f"Retrieved {len(df)} live bars for {symbol} from Polygon.io (verified live source)")
                return df
            
            logger.warning(f"No historical data from Polygon.io for {symbol}")
            return None
            
        except Exception as e:
            # Sanitize error message to prevent API key leakage
            error_msg = str(e).replace(self.api_key, '[REDACTED]') if self.api_key else str(e)
            logger.error(f"Failed to get historical data for {symbol} from Polygon.io: {error_msg}")
            # Return cached data if available
            if cache_key in self.ohlc_cache:
                logger.info(f"Using cached OHLC data for {symbol} due to error")
                return self.ohlc_cache[cache_key]['data']
            return None
    
    
    def get_available_pairs(self) -> List[str]:
        """Get list of available currency pairs"""
        return list(self.forex_mapping.keys())
    
    async def test_connection(self) -> bool:
        """Test connection to Polygon.io API"""
        try:
            # Test with a simple market status request
            endpoint = "/v1/marketstatus/now"
            response = await self._make_request(endpoint)
            data = response.json()
            return data.get('status') == 'OK'
        except:
            return False
    
    async def get_news(self, category: str = 'general', limit: int = 20) -> Optional[List[Dict[str, Any]]]:
        """
        Get financial news articles
        
        Note: Polygon.io provider is primarily for price data.
        News functionality should be handled by dedicated news providers.
        
        Args:
            category: News category ('general', 'forex', 'crypto', etc.)
            limit: Number of articles to retrieve
            
        Returns:
            Empty list - this provider doesn't handle news
        """
        logger.info(f"Polygon.io provider: News requests should use dedicated news providers")
        return []
    
    async def get_symbol_news(self, symbol: str, limit: int = 10) -> Optional[List[Dict[str, Any]]]:
        """
        Get news articles related to a specific symbol/ticker
        
        Note: Polygon.io provider is primarily for price data.
        News functionality should be handled by dedicated news providers.
        
        Args:
            symbol: Symbol to get news for (e.g., 'EURUSD', 'BTCUSD')
            limit: Number of articles to retrieve
            
        Returns:
            Empty list - this provider doesn't handle news
        """
        logger.info(f"Polygon.io provider: Symbol news requests for {symbol} should use dedicated news providers")
        return []