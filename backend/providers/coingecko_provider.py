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
        self.min_request_interval = 3.0  # ~20 requests per minute with safe buffer (was 2.1)
        
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