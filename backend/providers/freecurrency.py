"""
Free Currency API Provider - Real-time forex data
Uses FreeCurrencyAPI.com for live exchange rates
"""
import pandas as pd
import numpy as np
import httpx
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from pathlib import Path
import os
import time

from .base import BaseDataProvider
from ..logs.logger import get_logger

logger = get_logger(__name__)

class FreeCurrencyAPIProvider(BaseDataProvider):
    """Live forex data provider using FreeCurrencyAPI.com"""
    
    def __init__(self):
        self.base_url = "https://api.freecurrencyapi.com/v1"
        self.api_key = os.getenv('FREECURRENCY_API_KEY', None)  # Optional API key for higher limits
        self.cache_dir = Path("data/live")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_cache = {}
        self.last_update = {}
        
        # Shared async client for connection reuse
        self._client = None
        self._availability_checked = False
        self._is_api_available = False
        self._last_availability_check = 0
        
        # Major forex pairs to base currency mapping
        self.pair_mapping = {
            'EURUSD': ('EUR', 'USD'), 'GBPUSD': ('GBP', 'USD'), 'USDJPY': ('USD', 'JPY'),
            'AUDUSD': ('AUD', 'USD'), 'USDCAD': ('USD', 'CAD'), 'USDCHF': ('USD', 'CHF'),
            'NZDUSD': ('NZD', 'USD'), 'EURGBP': ('EUR', 'GBP'), 'EURJPY': ('EUR', 'JPY'),
            'GBPJPY': ('GBP', 'JPY'), 'AUDJPY': ('AUD', 'JPY'), 'CHFJPY': ('CHF', 'JPY'),
            'EURCHF': ('EUR', 'CHF'), 'GBPAUD': ('GBP', 'AUD'), 'AUDCAD': ('AUD', 'CAD')
        }
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create shared async client"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=10)
        return self._client
    
    async def _check_api_availability(self) -> bool:
        """Actually test if the API is working"""
        try:
            client = await self._get_client()
            params = {}
            if self.api_key:
                params['apikey'] = self.api_key
                
            response = await client.get(
                f"{self.base_url}/latest",
                params={**params, 'base_currency': 'USD'},
            )
            
            if response.status_code == 200:
                data = response.json()
                return 'data' in data
            else:
                logger.warning(f"FreeCurrency API check failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.warning(f"FreeCurrency API availability check failed: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if the API is available (cached result)"""
        # Check cache first (check every 5 minutes)
        now = time.time()
        if self._availability_checked and (now - self._last_availability_check) < 300:
            return self._is_api_available
            
        # Return False if we haven't checked or it's been too long
        # The actual check happens async in get_live_rates
        return False
    
    async def get_live_rates(self) -> dict:
        """Fetch latest exchange rates from FreeCurrencyAPI"""
        try:
            # Use API key if available, otherwise use free tier
            params = {}
            if self.api_key:
                params['apikey'] = self.api_key
            
            client = await self._get_client()
            # Get latest rates with USD as base
            response = await client.get(
                f"{self.base_url}/latest",
                params={**params, 'base_currency': 'USD'},
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data:
                    # Update availability cache on success
                    self._is_api_available = True
                    self._availability_checked = True
                    self._last_availability_check = time.time()
                    return data['data']
                else:
                    logger.error(f"Invalid API response format: {data}")
                    return {}
            elif response.status_code == 401:
                # API key required or invalid
                logger.warning("FreeCurrency API requires authentication (401 Unauthorized)")
                self._is_api_available = False
                self._availability_checked = True
                self._last_availability_check = time.time()
                return {}
            else:
                logger.warning(f"FreeCurrency API error: {response.status_code}")
                return {}
                    
        except Exception as e:
            logger.error(f"Error fetching live rates: {e}")
            return {}
    
    def calculate_cross_rate(self, base_rates: dict, from_currency: str, to_currency: str) -> float:
        """Calculate cross currency rate from USD base rates"""
        if from_currency == 'USD':
            return base_rates.get(to_currency, 1.0)
        elif to_currency == 'USD':
            return 1.0 / base_rates.get(from_currency, 1.0) if base_rates.get(from_currency) else 1.0
        else:
            # Cross rate calculation: FROM/TO = (FROM/USD) / (TO/USD)
            from_usd = base_rates.get(from_currency, 1.0)
            to_usd = base_rates.get(to_currency, 1.0)
            if from_usd > 0 and to_usd > 0:
                return from_usd / to_usd
            return 1.0
    
    async def get_ohlc_data(
        self, 
        symbol: str, 
        timeframe: str = "M1", 
        limit: int = 200
    ) -> Optional[pd.DataFrame]:
        """
        Get OHLC data for a forex pair
        Since this API provides rates, we'll create OHLC candles with slight variations
        """
        try:
            if symbol not in self.pair_mapping:
                logger.warning(f"Unsupported forex pair: {symbol}")
                return None
                
            from_currency, to_currency = self.pair_mapping[symbol]
            
            # Check cache freshness (update every minute for real-time)
            cache_key = symbol
            now = datetime.utcnow()
            
            if (cache_key in self.last_update and 
                (now - self.last_update[cache_key]).seconds < 60):
                # Use cached data if less than 1 minute old
                cached_file = self.cache_dir / f"{symbol}_live.csv"
                if cached_file.exists():
                    df = pd.read_csv(cached_file)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    return df.tail(limit)
            
            # Fetch fresh rates
            rates = await self.get_live_rates()
            if not rates:
                logger.warning(f"No rates available for {symbol}")
                return None
                
            # Calculate current rate
            current_rate = self.calculate_cross_rate(rates, from_currency, to_currency)
            
            # Load existing data or create new
            cached_file = self.cache_dir / f"{symbol}_live.csv"
            if cached_file.exists():
                existing_df = pd.read_csv(cached_file)
                existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
                last_price = existing_df['close'].iloc[-1]
                last_time = existing_df['timestamp'].iloc[-1]
            else:
                existing_df = pd.DataFrame()
                last_price = current_rate
                last_time = now - timedelta(minutes=200)  # Start 200 minutes ago
            
            # Create new minute candle
            current_minute = now.replace(second=0, microsecond=0)
            
            # Generate realistic OHLC from current rate
            # Add small random variations to simulate intraday movement
            volatility = 0.0001 if 'JPY' not in symbol else 0.01  # JPY pairs have different pip values
            
            # Simulate price movement from last price to current rate
            price_change = (current_rate - last_price) * np.random.uniform(0.8, 1.2)
            open_price = last_price + price_change * 0.3
            close_price = current_rate
            
            # Generate high/low with realistic spread
            spread = volatility * np.random.uniform(0.5, 2.0)
            high_price = max(open_price, close_price) + spread * np.random.uniform(0.3, 0.8)
            low_price = min(open_price, close_price) - spread * np.random.uniform(0.3, 0.8)
            
            # Create new candle data
            new_candle = {
                'timestamp': current_minute,
                'open': round(open_price, 5),
                'high': round(high_price, 5),
                'low': round(low_price, 5),
                'close': round(close_price, 5),
                'volume': np.random.randint(50, 200)
            }
            
            # Append to existing data
            if not existing_df.empty:
                # Only add if it's a new minute
                if existing_df['timestamp'].iloc[-1] < current_minute:
                    new_df = pd.concat([existing_df, pd.DataFrame([new_candle])], ignore_index=True)
                else:
                    # Update current minute candle
                    existing_df.loc[existing_df.index[-1], 'high'] = max(existing_df['high'].iloc[-1], high_price)
                    existing_df.loc[existing_df.index[-1], 'low'] = min(existing_df['low'].iloc[-1], low_price)
                    existing_df.loc[existing_df.index[-1], 'close'] = close_price
                    new_df = existing_df
            else:
                # Generate initial historical data
                historical_data = []
                current_time = current_minute - timedelta(minutes=limit-1)
                current_price = current_rate
                
                for i in range(limit):
                    # Simulate realistic price walk
                    price_change = np.random.normal(0, volatility)
                    open_p = current_price
                    close_p = current_price + price_change
                    
                    spread = volatility * np.random.uniform(0.5, 1.5)
                    high_p = max(open_p, close_p) + spread * np.random.uniform(0, 0.6)
                    low_p = min(open_p, close_p) - spread * np.random.uniform(0, 0.6)
                    
                    historical_data.append({
                        'timestamp': current_time + timedelta(minutes=i),
                        'open': round(open_p, 5),
                        'high': round(high_p, 5),
                        'low': round(low_p, 5),
                        'close': round(close_p, 5),
                        'volume': np.random.randint(50, 200)
                    })
                    
                    current_price = close_p
                
                new_df = pd.DataFrame(historical_data)
            
            # Keep only last 500 candles to manage file size
            new_df = new_df.tail(500)
            
            # Save to cache
            new_df.to_csv(cached_file, index=False)
            self.last_update[cache_key] = now
            
            logger.info(f"Updated live data for {symbol}: rate={current_rate:.5f}")
            return new_df.tail(limit)
            
        except Exception as e:
            logger.error(f"Error getting OHLC data for {symbol}: {e}")
            return None
    
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the latest price for a currency pair"""
        try:
            if symbol not in self.pair_mapping:
                return None
                
            from_currency, to_currency = self.pair_mapping[symbol]
            rates = await self.get_live_rates()
            if not rates:
                return None
                
            return self.calculate_cross_rate(rates, from_currency, to_currency)
            
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {e}")
            return None
    
    async def get_news(self, category: str = 'general', limit: int = 20) -> Optional[List[Dict[str, Any]]]:
        """
        Get financial news articles
        
        Note: FreeCurrencyAPI provider is for currency exchange rates only.
        News functionality should be handled by dedicated news providers.
        
        Args:
            category: News category ('general', 'forex', 'crypto', etc.)
            limit: Number of articles to retrieve
            
        Returns:
            Empty list - this provider doesn't handle news
        """
        logger.info(f"FreeCurrencyAPI provider: News requests should use dedicated news providers")
        return []
    
    async def get_symbol_news(self, symbol: str, limit: int = 10) -> Optional[List[Dict[str, Any]]]:
        """
        Get news articles related to a specific symbol/ticker
        
        Note: FreeCurrencyAPI provider is for currency exchange rates only.
        News functionality should be handled by dedicated news providers.
        
        Args:
            symbol: Symbol to get news for (e.g., 'EURUSD', 'BTCUSD')
            limit: Number of articles to retrieve
            
        Returns:
            Empty list - this provider doesn't handle news
        """
        logger.info(f"FreeCurrencyAPI provider: Symbol news requests for {symbol} should use dedicated news providers")
        return []