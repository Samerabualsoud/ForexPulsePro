import asyncio
import httpx
import pandas as pd
from typing import Optional
from datetime import datetime, timedelta
import structlog
import os

logger = structlog.get_logger(__name__)

class FinnhubProvider:
    """Finnhub.io forex data provider - Free tier with real-time data"""
    
    def __init__(self):
        self.api_key = os.getenv('FINNHUB_API_KEY')
        self.base_url = "https://finnhub.io/api/v1"
        self.session_timeout = 30
        
        # Forex symbol mapping (Finnhub format)
        self.symbol_mapping = {
            'EURUSD': 'OANDA:EUR_USD',
            'GBPUSD': 'OANDA:GBP_USD', 
            'USDJPY': 'OANDA:USD_JPY',
            'USDCHF': 'OANDA:USD_CHF',
            'AUDUSD': 'OANDA:AUD_USD',
            'USDCAD': 'OANDA:USD_CAD',
            'NZDUSD': 'OANDA:NZD_USD',
            'EURJPY': 'OANDA:EUR_JPY',
            'GBPJPY': 'OANDA:GBP_JPY',
            'EURGBP': 'OANDA:EUR_GBP',
            'AUDJPY': 'OANDA:AUD_JPY',
            'EURAUD': 'OANDA:EUR_AUD',
            'EURCAD': 'OANDA:EUR_CAD',
            'EURCHF': 'OANDA:EUR_CHF',
            'AUDCAD': 'OANDA:AUD_CAD'
        }
        
        logger.info(f"Finnhub provider initialized with API key: {'✓' if self.api_key else '✗'}")
    
    def is_available(self) -> bool:
        """Check if Finnhub API key is available"""
        return self.api_key is not None
    
    def _get_finnhub_symbol(self, symbol: str) -> str:
        """Convert standard forex symbol to Finnhub format"""
        return self.symbol_mapping.get(symbol.upper(), f'OANDA:{symbol[:3]}_{symbol[3:]}')
    
    async def get_current_price(self, symbol: str) -> Optional[dict]:
        """Get current forex price"""
        if not self.is_available():
            return None
            
        try:
            finnhub_symbol = self._get_finnhub_symbol(symbol)
            
            async with httpx.AsyncClient(timeout=self.session_timeout) as client:
                response = await client.get(
                    f"{self.base_url}/quote",
                    params={
                        'symbol': finnhub_symbol,
                        'token': self.api_key
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check if data is valid
                    if data.get('c', 0) > 0:  # 'c' is current price
                        return {
                            'symbol': symbol,
                            'price': data.get('c', 0),
                            'change': data.get('d', 0),
                            'change_percent': data.get('dp', 0),
                            'high': data.get('h', 0),
                            'low': data.get('l', 0),
                            'open': data.get('o', 0),
                            'previous_close': data.get('pc', 0),
                            'timestamp': int(datetime.now().timestamp())
                        }
                        
        except Exception as e:
            logger.warning(f"Finnhub current price error for {symbol}: {e}")
            
        return None
    
    async def get_ohlc_data(self, symbol: str, limit: int = 200) -> Optional[pd.DataFrame]:
        """Get OHLC forex data from Finnhub"""
        if not self.is_available():
            return None
            
        try:
            finnhub_symbol = self._get_finnhub_symbol(symbol)
            
            # Calculate date range (Finnhub uses Unix timestamps)
            end_time = datetime.now()
            start_time = end_time - timedelta(days=max(1, limit // 24))  # Rough estimation
            
            async with httpx.AsyncClient(timeout=self.session_timeout) as client:
                response = await client.get(
                    f"{self.base_url}/forex/candle",
                    params={
                        'symbol': finnhub_symbol,
                        'resolution': 'D',  # Daily candles
                        'from': int(start_time.timestamp()),
                        'to': int(end_time.timestamp()),
                        'token': self.api_key
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check if we have valid data
                    if data.get('s') == 'ok' and data.get('c'):
                        # Create DataFrame from Finnhub response
                        df = pd.DataFrame({
                            'time': pd.to_datetime(data['t'], unit='s'),
                            'open': data['o'],
                            'high': data['h'], 
                            'low': data['l'],
                            'close': data['c'],
                            'volume': data.get('v', [0] * len(data['c']))
                        })
                        
                        df = df.set_index('time').sort_index()
                        
                        # Limit to requested number of bars
                        if len(df) > limit:
                            df = df.tail(limit)
                            
                        logger.info(f"Retrieved {len(df)} bars for {symbol} from Finnhub")
                        return df
                    else:
                        logger.warning(f"No Finnhub data for {symbol}: {data.get('s', 'unknown error')}")
                        
        except Exception as e:
            logger.error(f"Finnhub OHLC error for {symbol}: {e}")
            
        return None
    
    async def get_forex_rates(self, base_currency: str = 'USD') -> Optional[dict]:
        """Get current forex exchange rates"""
        if not self.is_available():
            return None
            
        try:
            async with httpx.AsyncClient(timeout=self.session_timeout) as client:
                response = await client.get(
                    f"{self.base_url}/forex/rates",
                    params={
                        'base': base_currency,
                        'token': self.api_key
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if 'quote' in data:
                        return {
                            'base': base_currency,
                            'rates': data['quote'],
                            'timestamp': int(datetime.now().timestamp())
                        }
                        
        except Exception as e:
            logger.warning(f"Finnhub forex rates error: {e}")
            
        return None
    
    async def test_connection(self) -> bool:
        """Test Finnhub API connection"""
        try:
            test_price = await self.get_current_price('EURUSD')
            return test_price is not None
        except Exception as e:
            logger.error(f"Finnhub connection test failed: {e}")
            return False