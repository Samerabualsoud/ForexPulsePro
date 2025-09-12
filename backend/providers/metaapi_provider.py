import asyncio
import pandas as pd
from typing import Optional, Dict, Any
import httpx
from datetime import datetime, timedelta
import structlog

logger = structlog.get_logger(__name__)

class MetaApiProvider:
    """MetaApi cloud-based MT5 data provider for ACY Securities"""
    
    def __init__(self):
        self.base_url = "https://mt-client-api-v1.london.agiliumtrade.ai"
        self.token = None  # Will be set from environment variable
        self.account_id = None  # Will be set from environment variable
        self.is_connected = False
        
    def is_available(self) -> bool:
        """Check if MetaApi service is available"""
        return self.token is not None and self.account_id is not None
    
    async def connect(self, token: str, account_id: str) -> bool:
        """Connect to MetaApi service"""
        try:
            self.token = token
            self.account_id = account_id
            
            # Test connection
            headers = {"auth-token": self.token}
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/users/current/accounts/{self.account_id}/connection-status",
                    headers=headers
                )
                
                if response.status_code == 200:
                    self.is_connected = True
                    logger.info("MetaApi connection established successfully")
                    return True
                else:
                    logger.error(f"MetaApi connection failed: {response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error connecting to MetaApi: {e}")
            return False
    
    async def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get account information"""
        if not self.is_connected:
            return None
            
        try:
            headers = {"auth-token": self.token}
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/users/current/accounts/{self.account_id}/account-information",
                    headers=headers
                )
                
                if response.status_code == 200:
                    return response.json()
                    
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            
        return None
    
    async def get_ohlc_data(self, symbol: str, limit: int = 200) -> Optional[pd.DataFrame]:
        """Get OHLC data for symbol"""
        if not self.is_connected:
            return None
            
        try:
            headers = {"auth-token": self.token}
            
            # Calculate start time (limit bars ago)
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(minutes=limit)  # Assuming 1-minute bars
            
            params = {
                "symbol": symbol,
                "timeframe": "1m",
                "startTime": start_time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "limit": limit
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/users/current/accounts/{self.account_id}/historical-market-data/candles",
                    headers=headers,
                    params=params
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Convert to DataFrame
                    if data and len(data) > 0:
                        df = pd.DataFrame(data)
                        
                        # Rename columns to match expected format
                        df = df.rename(columns={
                            'brokerTime': 'time',
                            'o': 'open',
                            'h': 'high', 
                            'l': 'low',
                            'c': 'close',
                            'v': 'volume'
                        })
                        
                        # Convert time to datetime
                        df['time'] = pd.to_datetime(df['time'])
                        df = df.set_index('time')
                        
                        logger.info(f"Retrieved {len(df)} bars for {symbol} from MetaApi")
                        return df
                        
        except Exception as e:
            logger.error(f"Error getting OHLC data for {symbol}: {e}")
            
        return None
    
    async def get_live_price(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get current live price for symbol"""
        if not self.is_connected:
            return None
            
        try:
            headers = {"auth-token": self.token}
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/users/current/accounts/{self.account_id}/symbols/{symbol}/current-price",
                    headers=headers
                )
                
                if response.status_code == 200:
                    price_data = response.json()
                    return {
                        'bid': price_data.get('bid', 0),
                        'ask': price_data.get('ask', 0),
                        'spread': price_data.get('ask', 0) - price_data.get('bid', 0)
                    }
                    
        except Exception as e:
            logger.error(f"Error getting live price for {symbol}: {e}")
            
        return None
    
    async def place_order(self, symbol: str, volume: float, order_type: str, 
                         stop_loss: Optional[float] = None, 
                         take_profit: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Place a trading order"""
        if not self.is_connected:
            return None
            
        try:
            headers = {"auth-token": self.token}
            
            # Get current price
            price_info = await self.get_live_price(symbol)
            if not price_info:
                return None
                
            order_data = {
                "actionType": "ORDER_TYPE_BUY" if order_type.upper() == "BUY" else "ORDER_TYPE_SELL",
                "symbol": symbol,
                "volume": volume,
                "stopLoss": stop_loss,
                "takeProfit": take_profit,
                "comment": "Automated signal from Forex Dashboard"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/users/current/accounts/{self.account_id}/trade",
                    headers=headers,
                    json=order_data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Order placed successfully: {result}")
                    return result
                else:
                    logger.error(f"Order placement failed: {response.status_code} - {response.text}")
                    
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            
        return None