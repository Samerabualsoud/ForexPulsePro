"""
Polygon.io API Provider for Real Live Market Data
Provides real-time and historical market data for forex, stocks, and crypto including Bitcoin
"""

import os
import requests
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import time
from .base import BaseDataProvider

logger = logging.getLogger(__name__)

class PolygonProvider(BaseDataProvider):
    """Polygon.io API provider for real live market data"""
    
    def __init__(self):
        super().__init__()
        self.name = "Polygon.io"
        self.base_url = "https://api.polygon.io"
        
        # API key from environment (REQUIRED - no default for security)
        self.api_key = os.getenv('POLYGON_API_KEY')
        
        # Rate limiting (5 calls per minute for free tier)
        self.last_request_time = 0
        self.min_request_interval = 12.1  # ~5 requests per minute with buffer
        
        # Forex symbol mapping for Polygon.io
        self.forex_mapping = {
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
            # Crypto mapping
            'BTCUSD': 'X:BTCUSD',
            'ETHUSD': 'X:ETHUSD',
        }
        
        logger.info(f"Polygon.io provider initialized for real live market data")
        
    def is_available(self) -> bool:
        """Check if Polygon.io API is available"""
        return bool(self.api_key)
    
    def _rate_limit(self):
        """Enforce rate limiting for free tier"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            # Note: Using blocking sleep in async context for now
            # TODO: Replace with asyncio.sleep in production
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _get_polygon_symbol(self, pair: str) -> Optional[str]:
        """Convert standard pair to Polygon.io format"""
        return self.forex_mapping.get(pair)
    
    def _make_request(self, endpoint: str, params: Dict = None) -> requests.Response:
        """Make rate-limited request to Polygon.io API"""
        self._rate_limit()
        
        url = f"{self.base_url}{endpoint}"
        
        # Add API key to params
        if params is None:
            params = {}
        params['apiKey'] = self.api_key  # Fixed: Use correct parameter name
        
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            return response
            
        except requests.exceptions.RequestException as e:
            # Sanitize error message to prevent API key leakage
            error_msg = str(e).replace(self.api_key, '[REDACTED]') if self.api_key else str(e)
            logger.error(f"Polygon.io API request failed: {error_msg}")
            raise
    
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get current live price for currency pair"""
        if not self.is_available():
            return None
            
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
            response = self._make_request(endpoint)
            data = response.json()
            
            if data.get('status') == 'OK' and 'results' in data:
                results = data['results']
                # Calculate mid price from bid/ask
                bid = results.get('bid', 0)
                ask = results.get('ask', 0)
                
                if bid > 0 and ask > 0:
                    mid_price = (bid + ask) / 2
                    logger.info(f"Polygon.io live price for {symbol}: {mid_price}")
                    return mid_price
            
            logger.warning(f"No live price data from Polygon.io for {symbol}")
            return None
            
        except Exception as e:
            # Sanitize error message to prevent API key leakage
            error_msg = str(e).replace(self.api_key, '[REDACTED]') if self.api_key else str(e)
            logger.error(f"Failed to get live price for {symbol} from Polygon.io: {error_msg}")
            return None
    
    async def get_ohlc_data(self, symbol: str, timeframe: str = "H1", limit: int = 100) -> Optional[pd.DataFrame]:
        """Get historical OHLC data from Polygon.io"""
        if not self.is_available():
            return None
            
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
            
            multiplier, timespan = timeframe_mapping.get(timeframe, (1, 'hour'))  # Fixed: Now uses passed timeframe
            
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
            
            response = self._make_request(endpoint, params)
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
                
                logger.info(f"Retrieved {len(df)} live bars for {symbol} from Polygon.io")
                return df
            
            logger.warning(f"No historical data from Polygon.io for {symbol}")
            return None
            
        except Exception as e:
            # Sanitize error message to prevent API key leakage
            error_msg = str(e).replace(self.api_key, '[REDACTED]') if self.api_key else str(e)
            logger.error(f"Failed to get historical data for {symbol} from Polygon.io: {error_msg}")
            return None
    
    
    def get_available_pairs(self) -> List[str]:
        """Get list of available currency pairs"""
        return list(self.forex_mapping.keys())
    
    def test_connection(self) -> bool:
        """Test connection to Polygon.io API"""
        try:
            # Test with a simple market status request
            endpoint = "/v1/marketstatus/now"
            response = self._make_request(endpoint)
            data = response.json()
            return data.get('status') == 'OK'
        except:
            return False