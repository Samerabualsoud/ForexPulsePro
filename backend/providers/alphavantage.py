"""
Alpha Vantage Data Provider - Production Implementation
"""
import os
import requests
import pandas as pd
import asyncio
import httpx
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import time
import json
import random

from .base import BaseDataProvider
from ..logs.logger import get_logger

logger = get_logger(__name__)

class AlphaVantageProvider(BaseDataProvider):
    """Alpha Vantage API data provider with rate limiting and caching"""
    
    def __init__(self):
        self.api_key = os.getenv("ALPHAVANTAGE_KEY")
        self.base_url = "https://www.alphavantage.co/query"
        self.enabled = bool(self.api_key)
        
        # Rate limiting (Alpha Vantage allows 5 calls per minute for free tier)
        self.calls_per_minute = 5
        self.call_timestamps = []
        self._rate_limit_lock = asyncio.Lock()
        
        # Shared async client for connection reuse
        self._client = None
        
        # Caching
        self.cache_dir = Path("data/alphavantage")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_duration = timedelta(minutes=1)  # Cache for 1 minute
        self.data_cache = {}
        self.price_cache = {}
        
        if not self.enabled:
            logger.info("Alpha Vantage provider disabled - no API key provided")
        else:
            logger.info("Alpha Vantage provider initialized with rate limiting")
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create shared async client"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30)
        return self._client
    
    async def _check_and_wait_rate_limit(self) -> None:
        """Thread-safe rate limiting with token bucket algorithm"""
        async with self._rate_limit_lock:
            now = time.time()
            
            # Remove timestamps older than 1 minute (token bucket refill)
            self.call_timestamps = [ts for ts in self.call_timestamps if now - ts < 60]
            
            # If we're at the limit, wait until we can make a call
            while len(self.call_timestamps) >= self.calls_per_minute:
                # Calculate wait time until oldest call expires
                if self.call_timestamps:
                    oldest_call = min(self.call_timestamps)
                    wait_time = max(1, 60 - (now - oldest_call))
                else:
                    wait_time = 12
                
                logger.info(f"Alpha Vantage rate limit reached, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                
                # Refresh timestamps after waiting
                now = time.time()
                self.call_timestamps = [ts for ts in self.call_timestamps if now - ts < 60]
            
            # Record this call
            self.call_timestamps.append(now)
    
    def _get_cache_key(self, symbol: str, timeframe: str, data_type: str) -> str:
        """Generate cache key for data"""
        return f"{symbol}_{timeframe}_{data_type}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.data_cache:
            return False
        
        cached_data = self.data_cache[cache_key]
        if 'timestamp' not in cached_data:
            return False
            
        cache_time = cached_data['timestamp']
        return datetime.now() - cache_time < self.cache_duration
    
    async def get_ohlc_data(
        self, 
        symbol: str, 
        timeframe: str = "M1", 
        limit: int = 100,
        retry_count: int = 0
    ) -> Optional[pd.DataFrame]:
        """
        Get OHLC data from Alpha Vantage API with rate limiting and caching
        """
        if not self.enabled:
            return None
        
        # Check cache first
        cache_key = self._get_cache_key(symbol, timeframe, "ohlc")
        if self._is_cache_valid(cache_key):
            logger.debug(f"Using cached data for {symbol}")
            return self.data_cache[cache_key]['data'].tail(limit) if limit else self.data_cache[cache_key]['data']
        
        try:
            # Convert symbol format (EURUSD -> EUR/USD)
            if len(symbol) == 6:
                from_currency = symbol[:3]
                to_currency = symbol[3:]
            else:
                logger.error(f"Invalid symbol format: {symbol}")
                return None
            
            # Wait for rate limit if necessary
            await self._check_and_wait_rate_limit()
            
            # Map timeframe to Alpha Vantage intervals
            interval_map = {
                "M1": "1min",
                "M5": "5min", 
                "M15": "15min",
                "M30": "30min",
                "H1": "60min"
            }
            
            interval = interval_map.get(timeframe, "1min")
            
            params = {
                "function": "FX_INTRADAY",
                "from_symbol": from_currency,
                "to_symbol": to_currency,
                "interval": interval,
                "apikey": self.api_key,
                "outputsize": "full"  # Get more data points
            }
            
            # Make async request with shared client
            client = await self._get_client()
            response = await client.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Check for API errors
            if "Error Message" in data:
                logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                return None
            
            if "Note" in data:
                # Exponential backoff with jitter for rate limit notes
                if retry_count >= 3:  # Max 3 retries
                    logger.error(f"Alpha Vantage max retries exceeded: {data['Note']}")
                    return None
                
                base_delay = 15 * (2 ** retry_count)  # Exponential backoff
                jitter = random.uniform(0.8, 1.2)  # Add jitter
                delay = min(base_delay * jitter, 120)  # Cap at 2 minutes
                
                logger.warning(f"Alpha Vantage rate limit hit (retry {retry_count + 1}/3): {data['Note']}, waiting {delay:.1f}s")
                await asyncio.sleep(delay)
                
                # Increment retry count and recurse with updated count
                return await self.get_ohlc_data(symbol, timeframe, limit, retry_count + 1)
            
            if "Information" in data and "rate limit" in data["Information"].lower():
                logger.warning(f"Alpha Vantage rate limit: {data['Information']}")
                return None
            
            # Parse time series data
            time_series_key = f"Time Series FX ({interval})"
            if time_series_key not in data:
                logger.error(f"No time series data found for {symbol}. Available keys: {list(data.keys())}")
                return None
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            rows = []
            for timestamp_str, ohlc in time_series.items():
                try:
                    rows.append({
                        'timestamp': pd.to_datetime(timestamp_str),
                        'open': float(ohlc['1. open']),
                        'high': float(ohlc['2. high']),
                        'low': float(ohlc['3. low']),
                        'close': float(ohlc['4. close']),
                        'volume': 0  # Forex doesn't have volume in Alpha Vantage
                    })
                except (KeyError, ValueError) as e:
                    logger.warning(f"Skipping invalid data point for {timestamp_str}: {e}")
                    continue
            
            if not rows:
                logger.error(f"No valid data points found for {symbol}")
                return None
            
            df = pd.DataFrame(rows)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Cache the data
            self.data_cache[cache_key] = {
                'data': df,
                'timestamp': datetime.now()
            }
            
            # Save to disk cache as well
            cache_file = self.cache_dir / f"{cache_key}.json"
            try:
                cache_data = {
                    'data': df.to_dict('records'),
                    'timestamp': datetime.now().isoformat()
                }
                with open(cache_file, 'w') as f:
                    json.dump(cache_data, f)
            except Exception as e:
                logger.warning(f"Failed to save cache to disk: {e}")
            
            result_df = df.tail(limit) if limit else df
            logger.info(f"Retrieved {len(df)} bars for {symbol} from Alpha Vantage (returning {len(result_df)})")
            return result_df
            
        except httpx.RequestError as e:
            logger.error(f"Network error retrieving data for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving data for {symbol}: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if Alpha Vantage is configured and available"""
        return self.enabled
    
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price from Alpha Vantage with caching and rate limiting"""
        if not self.enabled:
            return None
        
        # Check cache first
        cache_key = f"{symbol}_latest_price"
        if cache_key in self.price_cache:
            cached_data = self.price_cache[cache_key]
            if datetime.now() - cached_data['timestamp'] < timedelta(seconds=30):
                return cached_data['price']
        
        try:
            # Convert symbol format
            if len(symbol) == 6:
                from_currency = symbol[:3]
                to_currency = symbol[3:]
            else:
                return None
            
            # Wait for rate limit if necessary
            await self._check_and_wait_rate_limit()
            
            params = {
                "function": "CURRENCY_EXCHANGE_RATE",
                "from_currency": from_currency,
                "to_currency": to_currency,
                "apikey": self.api_key
            }
            
            client = await self._get_client()
            response = await client.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "Error Message" in data:
                logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                return None
            
            if "Realtime Currency Exchange Rate" in data:
                rate_data = data["Realtime Currency Exchange Rate"]
                price = float(rate_data["5. Exchange Rate"])
                
                # Cache the price
                self.price_cache[cache_key] = {
                    'price': price,
                    'timestamp': datetime.now()
                }
                
                return price
            
            logger.warning(f"No exchange rate data found for {symbol}")
            return None
            
        except httpx.RequestError as e:
            logger.error(f"Network error getting latest price for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {e}")
            return None
