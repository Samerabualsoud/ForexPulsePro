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
import random
from .base import BaseDataProvider

logger = logging.getLogger(__name__)

class CoinGeckoProvider(BaseDataProvider):
    """CoinGecko API provider for free crypto market data"""
    
    def __init__(self):
        super().__init__()
        self.name = "CoinGecko"
        self.is_live_source = True  # CoinGecko provides real-time data
        self.base_url = "https://api.coingecko.com/api/v3"
        
        # Enhanced rate limiting with token bucket approach
        self._rate_limit_lock = asyncio.Lock()
        self.call_timestamps = []  # Track request timestamps
        self.calls_per_minute = 12  # Conservative limit (down from 30 to avoid 429)
        
        # Data caching to reduce API calls
        self.price_cache = {}  # Cache for price data
        self.ohlc_cache = {}   # Cache for OHLC data
        self.cache_duration = timedelta(seconds=45)  # 45-second cache
        
        # Batch optimization for multiple requests
        self._batch_queue = []  # Queue for batch processing
        self._batch_lock = asyncio.Lock()
        self._batch_cache = {}  # Global batch cache
        self._last_batch_time = 0
        
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
    
    async def _check_and_wait_rate_limit(self):
        """Advanced rate limiting with token bucket approach"""
        async with self._rate_limit_lock:
            now = time.time()
            
            # Remove timestamps older than 1 minute (token bucket refill)
            self.call_timestamps = [ts for ts in self.call_timestamps if now - ts < 60]
            
            # If we're at the limit, wait until we can make a call
            while len(self.call_timestamps) >= self.calls_per_minute:
                # Calculate wait time until oldest call expires
                if self.call_timestamps:
                    oldest_call = min(self.call_timestamps)
                    wait_time = max(2, 60 - (now - oldest_call) + 1)  # Extra 1s buffer
                else:
                    wait_time = 6
                
                logger.info(f"CoinGecko rate limit reached, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                
                # Refresh timestamps after waiting
                now = time.time()
                self.call_timestamps = [ts for ts in self.call_timestamps if now - ts < 60]
            
            # Record this call
            self.call_timestamps.append(now)
    
    def _get_coingecko_id(self, symbol: str) -> Optional[str]:
        """Convert standard crypto symbol to CoinGecko ID"""
        return self.crypto_mapping.get(symbol.upper())
    
    async def _make_request_with_retry(self, endpoint: str, params: Dict = None, max_retries: int = 3) -> Optional[httpx.Response]:
        """Make rate-limited request with exponential backoff retry"""
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(max_retries + 1):
            try:
                await self._check_and_wait_rate_limit()
                
                async with httpx.AsyncClient(timeout=20.0) as client:
                    response = await client.get(url, params=params or {})
                    
                    if response.status_code == 429:
                        # Rate limited - implement exponential backoff
                        if attempt < max_retries:
                            backoff_time = (2 ** attempt) * 5 + random.uniform(1, 3)  # 5-8s, 10-13s, 20-23s
                            logger.warning(f"CoinGecko 429 rate limit hit, attempt {attempt + 1}/{max_retries + 1}, waiting {backoff_time:.1f}s")
                            await asyncio.sleep(backoff_time)
                            continue
                        else:
                            logger.error(f"CoinGecko rate limit exceeded after {max_retries + 1} attempts")
                            return None
                    
                    response.raise_for_status()
                    return response
                    
            except httpx.TimeoutException:
                if attempt < max_retries:
                    wait_time = 2 ** attempt + random.uniform(0.5, 1.5)
                    logger.warning(f"CoinGecko timeout, retrying in {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"CoinGecko timeout after {max_retries + 1} attempts")
                    return None
                    
            except httpx.HTTPError as e:
                logger.error(f"CoinGecko API request failed: {e}")
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)
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
        """Get current live price for cryptocurrency with intelligent batching"""
        if not self.is_available():
            return None
        
        # Check cache first
        cache_key = self._get_cache_key(symbol, "price")
        if self._is_cache_valid(cache_key, self.price_cache):
            logger.debug(f"Using cached price for {symbol}")
            return self.price_cache[cache_key]['price']
        
        # Check batch cache
        if cache_key in self._batch_cache:
            batch_data = self._batch_cache[cache_key]
            if datetime.now() - batch_data['timestamp'] < self.cache_duration:
                logger.debug(f"Using batch cached price for {symbol}")
                return batch_data['price']
        
        # Try batch request first (more efficient)
        result = await self._get_price_from_batch(symbol)
        if result is not None:
            return result
        
        # Fallback to individual request
        return await self._get_individual_price(symbol)
    
    async def _get_price_from_batch(self, symbol: str) -> Optional[float]:
        """Get price using batch request optimization"""
        try:
            # Get all crypto symbols we need (simulate typical request pattern)
            crypto_symbols = ['BTCUSD', 'ETHUSD', 'ADAUSD', 'DOGEUSD', 'SOLUSD', 'BNBUSD', 'XRPUSD', 'MATICUSD']
            
            # Build batch request for all crypto symbols
            coingecko_ids = []
            symbol_mapping = {}
            
            for sym in crypto_symbols:
                coingecko_id = self._get_coingecko_id(sym)
                if coingecko_id:
                    coingecko_ids.append(coingecko_id)
                    symbol_mapping[coingecko_id] = sym
            
            if not coingecko_ids:
                return None
            
            # Make single batch request
            endpoint = "/simple/price"
            params = {
                'ids': ','.join(coingecko_ids),  # Batch all IDs in one request
                'vs_currencies': 'usd',
                'include_24hr_change': 'true',
                'include_last_updated_at': 'true'
            }
            
            response = await self._make_request_with_retry(endpoint, params)
            if not response:
                return None
            
            data = response.json()
            
            # Cache all results from batch
            current_time = datetime.now()
            for coingecko_id, sym in symbol_mapping.items():
                if coingecko_id in data and 'usd' in data[coingecko_id]:
                    price = float(data[coingecko_id]['usd'])
                    
                    # Cache in both individual and batch caches
                    cache_key = self._get_cache_key(sym, "price")
                    self.price_cache[cache_key] = {
                        'price': price,
                        'timestamp': current_time
                    }
                    self._batch_cache[cache_key] = {
                        'price': price,
                        'timestamp': current_time
                    }
                    
                    logger.info(f"CoinGecko batch price for {sym}: ${price}")
            
            # Return price for requested symbol
            request_cache_key = self._get_cache_key(symbol, "price")
            if request_cache_key in self.price_cache:
                return self.price_cache[request_cache_key]['price']
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get batch price for {symbol} from CoinGecko: {e}")
            return None
    
    async def _get_individual_price(self, symbol: str) -> Optional[float]:
        """Fallback to individual price request"""
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
            
            response = await self._make_request_with_retry(endpoint, params)
            if not response:
                logger.error(f"Failed to get price for {symbol} from CoinGecko after retries")
                # Return cached data if available, even if expired
                cache_key = self._get_cache_key(symbol, "price")
                if cache_key in self.price_cache:
                    logger.info(f"Using expired cached price for {symbol} due to API failure")
                    return self.price_cache[cache_key]['price']
                return None
            
            data = response.json()
            
            if coingecko_id in data and 'usd' in data[coingecko_id]:
                price = float(data[coingecko_id]['usd'])
                
                # Cache the result
                cache_key = self._get_cache_key(symbol, "price")
                self.price_cache[cache_key] = {
                    'price': price,
                    'timestamp': datetime.now()
                }
                
                logger.info(f"CoinGecko individual price for {symbol}: ${price}")
                return price
            
            logger.warning(f"No live price data from CoinGecko for {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get individual price for {symbol} from CoinGecko: {e}")
            # Return cached data if available
            cache_key = self._get_cache_key(symbol, "price")
            if cache_key in self.price_cache:
                logger.info(f"Using cached price for {symbol} due to error")
                return self.price_cache[cache_key]['price']
            return None
    
    # REMOVED DUPLICATE METHOD - Using the improved cached version below
    
    async def get_ohlc_data(self, symbol: str, timeframe: str = "H1", limit: int = 100) -> Optional[pd.DataFrame]:
        """Get synthetic OHLC data from CoinGecko current prices with caching"""
        if not self.is_available():
            return None
        
        # Check cache first
        cache_key = self._get_cache_key(f"{symbol}_{timeframe}_{limit}", "ohlc")
        if self._is_cache_valid(cache_key, self.ohlc_cache):
            logger.debug(f"Using cached OHLC data for {symbol}")
            return self.ohlc_cache[cache_key]['data']
            
        try:
            coingecko_id = self._get_coingecko_id(symbol)
            if not coingecko_id:
                return None
            
            # Use batching optimization for OHLC data too
            df = await self._get_ohlc_with_batching(symbol, timeframe, limit)
            if df is not None:
                # Cache the result
                self.ohlc_cache[cache_key] = {
                    'data': df,
                    'timestamp': datetime.now()
                }
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol} from CoinGecko: {e}")
            # Return cached data if available
            if cache_key in self.ohlc_cache:
                logger.info(f"Using cached OHLC data for {symbol} due to error")
                return self.ohlc_cache[cache_key]['data']
            return None
    
    async def _get_ohlc_with_batching(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        """Generate OHLC data using batching optimization"""
        try:
            # First try to get the current price from batch cache or batch request
            current_price = await self.get_latest_price(symbol)
            if not current_price:
                return None
            
            # Generate synthetic OHLC data using the batched price
            return self._generate_synthetic_ohlc_from_price(current_price, symbol, timeframe, limit)
            
        except Exception as e:
            logger.error(f"Failed to get OHLC with batching for {symbol}: {e}")
            return None
    
    def _generate_synthetic_ohlc_from_price(self, current_price: float, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        """Generate synthetic OHLC data from a known current price (no API calls)"""
        try:
            # Validate current price is positive
            if current_price <= 0:
                logger.error(f"Invalid current price for {symbol}: ${current_price}")
                return None
            
            # Generate synthetic historical data with improved algorithm
            timeframe_minutes = {
                'M1': 1, 'M5': 5, 'M15': 15, 'H1': 60, 'H4': 240, 'D1': 1440
            }
            
            minutes_per_bar = timeframe_minutes.get(timeframe, 60)
            
            # Create realistic price variations based on crypto volatility
            # Use conservative volatility to prevent extreme price movements
            daily_volatility = 0.03  # 3% daily volatility for crypto (reduced from 5%)
            bar_volatility = daily_volatility * (minutes_per_bar / 1440) ** 0.5  # Scale by timeframe
            
            # Cap volatility to prevent extreme movements
            bar_volatility = min(bar_volatility, 0.02)  # Max 2% per bar
            
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
            
            # Generate synthetic price data with controlled algorithm
            df_data = []
            
            # Use numpy random for reproducible synthetic data
            np.random.seed(int(current_price * 1000) % 10000)  # Seed based on price for consistency
            
            # Generate price variations around the current price
            price_variations = np.random.normal(0, bar_volatility, limit)
            
            for i, timestamp in enumerate(timestamps):
                # Use simple, safe price generation around current price
                # Generate a small percentage variation (±1% max)
                variation_pct = price_variations[i] * 0.01  # Limit to ±1%
                variation_pct = max(-0.01, min(0.01, variation_pct))  # Clamp to ±1%
                
                if i == len(timestamps) - 1:
                    # Last bar should end at current price
                    close_p = current_price
                    open_p = current_price * (1 + variation_pct * 0.5)
                else:
                    # Generate close price with small variation
                    close_p = current_price * (1 + variation_pct)
                    open_p = current_price * (1 + variation_pct * 0.8)  # Slightly different open
                
                # Ensure prices are always positive and reasonable (never less than 80% of current price)
                min_price = current_price * 0.8
                close_p = max(close_p, min_price)
                open_p = max(open_p, min_price)
                
                # Generate high and low with simple logic
                high_base = max(open_p, close_p)
                low_base = min(open_p, close_p)
                
                # Add small range for high/low (0.1% to 0.5% of current price)
                price_range = current_price * np.random.uniform(0.001, 0.005)  # 0.1% to 0.5%
                
                high_p = high_base + price_range * np.random.uniform(0.5, 1.0)
                low_p = low_base - price_range * np.random.uniform(0.5, 1.0)
                
                # Ensure high/low boundaries
                high_p = max(high_p, open_p, close_p, min_price)
                low_p = max(low_p, min_price)  # Low must be positive
                low_p = min(low_p, open_p, close_p)  # Low must be below open/close
                
                # Final safety check - ensure all values are positive and reasonable
                open_p = max(open_p, current_price * 0.8)
                high_p = max(high_p, current_price * 0.8)
                low_p = max(low_p, current_price * 0.8)
                close_p = max(close_p, current_price * 0.8)
                
                # Round to appropriate precision based on price level
                if current_price < 1:
                    precision = 6
                elif current_price < 100:
                    precision = 4
                else:
                    precision = 2
                
                df_data.append({
                    'timestamp': timestamp,
                    'open': round(open_p, precision),
                    'high': round(high_p, precision),
                    'low': round(low_p, precision),
                    'close': round(close_p, precision),
                    'volume': np.random.randint(100000, 2000000)  # Crypto volume simulation
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            # Add metadata for validation
            df = self._add_metadata_to_dataframe(
                df,
                symbol,
                data_source=self.name,
                last_updated=datetime.now(timezone.utc).isoformat()
            )
            
            logger.info(f"Generated {len(df)} synthetic H1 crypto bars for {symbol} from CoinGecko batch price: ${current_price}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to generate synthetic OHLC for {symbol}: {e}")
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
            
            response = await self._make_request_with_retry(endpoint, params)
            if not response:
                logger.error(f"Failed to get synthetic OHLC for {symbol}")
                return None
                
            data = response.json()
            
            if coingecko_id not in data or 'usd' not in data[coingecko_id]:
                logger.warning(f"No current price data from CoinGecko for {symbol}")
                return None
            
            current_price = float(data[coingecko_id]['usd'])
            change_24h = data[coingecko_id].get('usd_24h_change', 0)
            
            # Validate current price is positive
            if current_price <= 0:
                logger.error(f"Invalid current price for {symbol}: ${current_price}")
                return None
            
            # Generate synthetic historical data with improved algorithm
            timeframe_minutes = {
                'M1': 1, 'M5': 5, 'M15': 15, 'H1': 60, 'H4': 240, 'D1': 1440
            }
            
            minutes_per_bar = timeframe_minutes.get(timeframe, 60)
            
            # Create realistic price variations based on crypto volatility
            # Use conservative volatility to prevent extreme price movements
            daily_volatility = 0.03  # 3% daily volatility for crypto (reduced from 5%)
            bar_volatility = daily_volatility * (minutes_per_bar / 1440) ** 0.5  # Scale by timeframe
            
            # Cap volatility to prevent extreme movements
            bar_volatility = min(bar_volatility, 0.02)  # Max 2% per bar
            
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
            
            # Generate synthetic price data with controlled algorithm
            df_data = []
            
            # Use numpy random for reproducible synthetic data
            np.random.seed(int(current_price * 1000) % 10000)  # Seed based on price for consistency
            
            # Generate price variations around the current price
            price_variations = np.random.normal(0, bar_volatility, limit)
            
            for i, timestamp in enumerate(timestamps):
                # Use simple, safe price generation around current price
                # Generate a small percentage variation (±1% max)
                variation_pct = price_variations[i] * 0.01  # Limit to ±1%
                variation_pct = max(-0.01, min(0.01, variation_pct))  # Clamp to ±1%
                
                if i == len(timestamps) - 1:
                    # Last bar should end at current price
                    close_p = current_price
                    open_p = current_price * (1 + variation_pct * 0.5)
                else:
                    # Generate close price with small variation
                    close_p = current_price * (1 + variation_pct)
                    open_p = current_price * (1 + variation_pct * 0.8)  # Slightly different open
                
                # Ensure prices are always positive and reasonable (never less than 80% of current price)
                min_price = current_price * 0.8
                close_p = max(close_p, min_price)
                open_p = max(open_p, min_price)
                
                # Generate high and low with simple logic
                high_base = max(open_p, close_p)
                low_base = min(open_p, close_p)
                
                # Add small range for high/low (0.1% to 0.5% of current price)
                price_range = current_price * np.random.uniform(0.001, 0.005)  # 0.1% to 0.5%
                
                high_p = high_base + price_range * np.random.uniform(0.5, 1.0)
                low_p = low_base - price_range * np.random.uniform(0.5, 1.0)
                
                # Ensure high/low boundaries
                high_p = max(high_p, open_p, close_p, min_price)
                low_p = max(low_p, min_price)  # Low must be positive
                low_p = min(low_p, open_p, close_p)  # Low must be below open/close
                
                # Final safety check - ensure all values are positive and reasonable
                open_p = max(open_p, current_price * 0.8)
                high_p = max(high_p, current_price * 0.8)
                low_p = max(low_p, current_price * 0.8)
                close_p = max(close_p, current_price * 0.8)
                
                # Round to appropriate precision based on price level
                if current_price < 1:
                    precision = 6
                elif current_price < 100:
                    precision = 4
                else:
                    precision = 2
                
                df_data.append({
                    'timestamp': timestamp,
                    'open': round(open_p, precision),
                    'high': round(high_p, precision),
                    'low': round(low_p, precision),
                    'close': round(close_p, precision),
                    'volume': np.random.randint(100000, 2000000)  # Crypto volume simulation
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            # Validate all OHLC data is positive before returning
            if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
                logger.error(f"Generated invalid OHLC data for {symbol} - contains zero/negative values")
                return None
                
            # Validate OHLC relationships
            invalid_bars = (
                (df['high'] < df[['open', 'close']].max(axis=1)) |
                (df['low'] > df[['open', 'close']].min(axis=1))
            )
            
            if invalid_bars.any():
                logger.warning(f"Generated {invalid_bars.sum()} invalid OHLC relationships for {symbol} - correcting...")
                # Fix invalid relationships
                for idx in df[invalid_bars].index:
                    df.loc[idx, 'high'] = max(df.loc[idx, 'open'], df.loc[idx, 'close'], df.loc[idx, 'high'])
                    df.loc[idx, 'low'] = min(df.loc[idx, 'open'], df.loc[idx, 'close'], df.loc[idx, 'low'])
            
            # Add metadata for real-time validation
            df = self._add_metadata_to_dataframe(
                df, 
                symbol, 
                data_source=self.name,
                last_updated=datetime.now(timezone.utc).isoformat()
            )
            
            self._log_data_fetch(symbol, True, len(df))
            logger.info(f"Generated {len(df)} valid synthetic {timeframe} crypto bars for {symbol} from CoinGecko current price: ${current_price}")
            logger.debug(f"OHLC sample for {symbol}: O={df.iloc[-1]['open']}, H={df.iloc[-1]['high']}, L={df.iloc[-1]['low']}, C={df.iloc[-1]['close']}")
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