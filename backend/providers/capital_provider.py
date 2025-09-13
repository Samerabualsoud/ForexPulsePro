"""
Capital.com API Provider for Forex Signal Dashboard
Provides real-time market data for forex pairs and Bitcoin using Capital.com API
"""

import os
import json
import time
import requests
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import hashlib
import base64
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from .base import BaseDataProvider

logger = logging.getLogger(__name__)

class CapitalProvider(BaseDataProvider):
    """Capital.com API provider for real-time market data"""
    
    def __init__(self):
        super().__init__()
        self.name = "Capital.com"
        # Use production Capital.com API for real live market data
        self.base_url = "https://api-capital.backend-capital.com"
        
        # API credentials from environment
        self.api_key = os.getenv('CAPITAL_API_KEY')
        self.username = os.getenv('CAPITAL_USERNAME')
        self.password = os.getenv('CAPITAL_PASSWORD')
        
        # Session tokens
        self.cst_token = None
        self.security_token = None
        self.session_expires = None
        
        # Available instruments cache
        self.instruments_cache = {}
        self.cache_expires = None
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 10 requests per second max
        
        logger.info(f"Capital.com provider initialized: {self.base_url}")
        
    def is_available(self) -> bool:
        """Check if Capital.com API is available with credentials"""
        if not all([self.api_key, self.username, self.password]):
            logger.warning("Capital.com API credentials not configured")
            return False
        return True
    
    def _rate_limit(self):
        """Enforce rate limiting (10 requests per second max)"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, method: str, endpoint: str, headers: Optional[Dict] = None, data: Optional[Dict] = None) -> requests.Response:
        """Make rate-limited request to Capital.com API"""
        self._rate_limit()
        
        url = f"{self.base_url}{endpoint}"
        default_headers = {
            'Content-Type': 'application/json',
            'X-CAP-API-KEY': self.api_key
        }
        
        if headers:
            default_headers.update(headers)
            
        if self.cst_token and self.security_token:
            default_headers.update({
                'CST': self.cst_token,
                'X-SECURITY-TOKEN': self.security_token
            })
        
        try:
            if method.upper() == 'GET':
                response = requests.get(url, headers=default_headers, timeout=10)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=default_headers, json=data, timeout=10)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
                
            response.raise_for_status()
            return response
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Capital.com API request failed: {e}")
            raise
    
    def _create_session(self) -> bool:
        """Create new authenticated session with Capital.com"""
        try:
            # Use simple authentication (password in plain text)
            # For production, consider implementing encrypted password method
            auth_data = {
                "identifier": self.username,
                "password": self.password,
                "encryptedPassword": False
            }
            
            response = self._make_request('POST', '/api/v1/session', data=auth_data)
            
            # Extract session tokens from headers
            self.cst_token = response.headers.get('CST')
            self.security_token = response.headers.get('X-SECURITY-TOKEN')
            
            if self.cst_token and self.security_token:
                self.session_expires = datetime.now() + timedelta(minutes=9)  # 9 min buffer
                logger.info("Capital.com session created successfully")
                return True
            else:
                logger.error("Failed to get session tokens from Capital.com")
                return False
                
        except Exception as e:
            logger.error(f"Failed to create Capital.com session: {e}")
            return False
    
    def _ensure_session(self) -> bool:
        """Ensure we have a valid session"""
        if not self.session_expires or datetime.now() >= self.session_expires:
            return self._create_session()
        return True
    
    def _get_epic_for_pair(self, pair: str) -> Optional[str]:
        """Get Capital.com epic identifier for currency pair"""
        # Common mappings for major pairs
        epic_mapping = {
            'EURUSD': 'CS.D.EURUSD.CFD.IP',
            'GBPUSD': 'CS.D.GBPUSD.CFD.IP', 
            'USDJPY': 'CS.D.USDJPY.CFD.IP',
            'AUDUSD': 'CS.D.AUDUSD.CFD.IP',
            'USDCAD': 'CS.D.USDCAD.CFD.IP',
            'USDCHF': 'CS.D.USDCHF.CFD.IP',
            'NZDUSD': 'CS.D.NZDUSD.CFD.IP',
            'EURGBP': 'CS.D.EURGBP.CFD.IP',
            'EURJPY': 'CS.D.EURJPY.CFD.IP',
            'GBPJPY': 'CS.D.GBPJPY.CFD.IP',
            'BTCUSD': 'CS.D.BITCOIN.CFD.IP',  # Bitcoin support
            'ETHUSD': 'CS.D.ETHUSD.CFD.IP',   # Ethereum for future
        }
        
        return epic_mapping.get(pair)
    
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the latest price for a symbol (required by BaseDataProvider)"""
        return self.get_current_price(symbol)
    
    async def get_ohlc_data(self, symbol: str, timeframe: str = "M1", limit: int = 100) -> Optional[pd.DataFrame]:
        """Get OHLC data for a symbol (required by BaseDataProvider)"""
        # Convert BaseDataProvider timeframe format to Capital.com format
        timeframe_mapping = {
            'M1': '1M',
            'M5': '5M', 
            'M15': '15M',
            'H1': '1H',
            'H4': '4H',
            'D1': '1D'
        }
        capital_timeframe = timeframe_mapping.get(timeframe, '1H')
        return self.get_historical_data(symbol, capital_timeframe, limit)

    def get_current_price(self, pair: str) -> Optional[float]:
        """Get current price for currency pair from Capital.com"""
        if not self.is_available():
            return None
            
        try:
            if not self._ensure_session():
                return None
                
            epic = self._get_epic_for_pair(pair)
            if not epic:
                logger.warning(f"No epic mapping found for pair: {pair}")
                return None
            
            # Get current market prices
            response = self._make_request('GET', f'/api/v1/markets/{epic}')
            market_data = response.json()
            
            # Extract current bid/ask prices
            if 'snapshot' in market_data:
                snapshot = market_data['snapshot']
                bid = float(snapshot.get('bid', 0))
                ask = float(snapshot.get('offer', 0))  # Capital.com uses 'offer' for ask
                
                if bid > 0 and ask > 0:
                    # Return mid price
                    current_price = (bid + ask) / 2
                    logger.info(f"Capital.com {pair}: {current_price}")
                    return current_price
            
            logger.warning(f"Invalid price data from Capital.com for {pair}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get price for {pair} from Capital.com: {e}")
            return None
    
    def get_historical_data(self, pair: str, timeframe: str = "1H", limit: int = 100) -> Optional[pd.DataFrame]:
        """Get historical OHLC data from Capital.com"""
        if not self.is_available():
            return None
            
        try:
            if not self._ensure_session():
                return None
                
            epic = self._get_epic_for_pair(pair)
            if not epic:
                return None
            
            # Capital.com resolution mapping
            resolution_mapping = {
                '1M': 'MINUTE',
                '5M': 'MINUTE_5',
                '15M': 'MINUTE_15',
                '1H': 'HOUR',
                '4H': 'HOUR_4',
                '1D': 'DAY'
            }
            
            resolution = resolution_mapping.get(timeframe, 'HOUR')
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(hours=limit)
            
            params = {
                'resolution': resolution,
                'from': start_date.strftime('%Y-%m-%dT%H:%M:%S'),
                'to': end_date.strftime('%Y-%m-%dT%H:%M:%S')
            }
            
            # Get historical prices
            response = self._make_request('GET', f'/api/v1/prices/{epic}', headers={'params': json.dumps(params)})
            data = response.json()
            
            if 'prices' in data and data['prices']:
                # Convert to DataFrame
                prices = data['prices']
                df_data = []
                
                for price in prices:
                    df_data.append({
                        'timestamp': pd.to_datetime(price['snapshotTime']),
                        'open': float(price['openPrice']['bid']),
                        'high': float(price['highPrice']['bid']), 
                        'low': float(price['lowPrice']['bid']),
                        'close': float(price['closePrice']['bid']),
                        'volume': 0  # Capital.com doesn't provide volume for forex
                    })
                
                df = pd.DataFrame(df_data)
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                
                logger.info(f"Retrieved {len(df)} historical bars for {pair} from Capital.com")
                return df
            
            logger.warning(f"No historical data available for {pair} from Capital.com")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get historical data for {pair} from Capital.com: {e}")
            return None
    
    def get_available_pairs(self) -> List[str]:
        """Get list of available currency pairs"""
        return [
            'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 
            'NZDUSD', 'EURGBP', 'EURJPY', 'GBPJPY', 'BTCUSD'  # Added Bitcoin
        ]
    
    def test_connection(self) -> bool:
        """Test connection to Capital.com API"""
        if not self.is_available():
            return False
            
        try:
            # Test ping endpoint
            response = self._make_request('GET', '/api/v1/ping')
            return response.status_code == 200
        except:
            return False