"""
MT5 Bridge Market Data Provider
Gets OHLC data from MT5 Bridge Service for real ACY Securities market data
"""
import pandas as pd
import httpx
import asyncio
import os
from datetime import datetime
from typing import Optional

from ..logs.logger import get_logger

logger = get_logger(__name__)

class MT5DataProvider:
    """MT5 Bridge market data provider"""
    
    def __init__(self):
        self.bridge_url = os.getenv('MT5_BRIDGE_URL', '')
        self.bridge_secret = os.getenv('MT5_BRIDGE_SECRET', '')
        self.timeout = 15.0
        self.retry_attempts = 2
        
        self._available = bool(self.bridge_url and self.bridge_secret)
        
        if not self._available:
            logger.warning("MT5 bridge not configured - MT5_BRIDGE_URL or MT5_BRIDGE_SECRET missing")
        else:
            logger.info(f"MT5 data provider initialized: {self.bridge_url}")
    
    def is_available(self) -> bool:
        """Check if MT5 bridge is available"""
        return self._available
    
    async def get_ohlc_data(self, symbol: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Get OHLC data from MT5 bridge"""
        if not self.is_available():
            return None
        
        try:
            # Get data from MT5 bridge symbol endpoint
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                headers = {
                    'Authorization': f'Bearer {self.bridge_secret}',
                    'Content-Type': 'application/json'
                }
                
                # Try to get OHLC data - the bridge should have an endpoint for this
                response = await client.get(
                    f"{self.bridge_url}/api/ohlc/{symbol}?count={limit}",
                    headers=headers
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'data' in data and data['data']:
                        # Convert to DataFrame
                        df = pd.DataFrame(data['data'])
                        
                        # Ensure we have the required columns
                        if all(col in df.columns for col in ['time', 'open', 'high', 'low', 'close']):
                            # Convert time to datetime
                            df['time'] = pd.to_datetime(df['time'])
                            df.set_index('time', inplace=True)
                            
                            # Ensure numeric types
                            for col in ['open', 'high', 'low', 'close']:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                            
                            # Add volume if not present
                            if 'tick_volume' not in df.columns:
                                df['tick_volume'] = 100  # Default volume
                            
                            logger.info(f"Retrieved {len(df)} bars of MT5 data for {symbol} from ACY Securities")
                            return df
                        else:
                            logger.warning(f"MT5 data for {symbol} missing required OHLC columns")
                            return None
                    else:
                        logger.warning(f"No MT5 data returned for {symbol}")
                        return None
                        
                elif response.status_code == 404:
                    logger.warning(f"Symbol {symbol} not found on MT5 bridge")
                    return None
                elif response.status_code == 503:
                    logger.warning(f"MT5 bridge service unavailable")
                    return None
                else:
                    logger.error(f"MT5 bridge error {response.status_code}: {response.text}")
                    return None
                    
        except httpx.TimeoutException:
            logger.warning(f"MT5 bridge timeout for {symbol}")
            return None
        except Exception as e:
            logger.error(f"MT5 bridge connection error for {symbol}: {e}")
            return None
    
    async def health_check(self) -> bool:
        """Check if MT5 bridge is healthy"""
        if not self.is_available():
            return False
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.bridge_url}/health")
                
                if response.status_code == 200:
                    health_data = response.json()
                    return health_data.get('status') == 'healthy' and health_data.get('mt5_connected', False)
                else:
                    return False
                    
        except Exception as e:
            logger.error(f"MT5 bridge health check failed: {e}")
            return False
    
    async def get_current_price(self, symbol: str) -> Optional[dict]:
        """Get current bid/ask price for symbol"""
        if not self.is_available():
            return None
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                headers = {
                    'Authorization': f'Bearer {self.bridge_secret}',
                    'Content-Type': 'application/json'
                }
                
                response = await client.get(
                    f"{self.bridge_url}/symbol/{symbol}",
                    headers=headers
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        'symbol': symbol,
                        'bid': data.get('bid', 0),
                        'ask': data.get('ask', 0),
                        'spread': data.get('spread', 0),
                        'time': datetime.now().isoformat()
                    }
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None