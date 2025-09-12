import asyncio
import pandas as pd
from typing import Optional
from datetime import datetime, timedelta
import structlog
import os
import finnhub

logger = structlog.get_logger(__name__)

class FinnhubProvider:
    """Finnhub.io forex data provider - Free tier with real-time data"""
    
    def __init__(self):
        self.api_key = os.getenv('FINNHUB_API_KEY')
        self.client = None
        if self.api_key:
            try:
                self.client = finnhub.Client(api_key=self.api_key)
                logger.info(f"Finnhub client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Finnhub client: {e}")
        else:
            logger.info("No Finnhub API key found")
        
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
        """Check if Finnhub client is available"""
        return self.client is not None
    
    def _get_finnhub_symbol(self, symbol: str) -> str:
        """Convert standard forex symbol to Finnhub format"""
        return self.symbol_mapping.get(symbol.upper(), f'OANDA:{symbol[:3]}_{symbol[3:]}')
    
    async def get_current_price(self, symbol: str) -> Optional[dict]:
        """Get current forex price"""
        if not self.is_available():
            return None
            
        try:
            finnhub_symbol = self._get_finnhub_symbol(symbol)
            
            # Run in thread pool since finnhub client is synchronous
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None, self.client.quote, finnhub_symbol
            )
            
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
            start_time = end_time - timedelta(days=max(30, limit))  # Get more days for better data
            
            # Run in thread pool since finnhub client is synchronous
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None,
                self.client.forex_candles,
                finnhub_symbol,
                'D',  # Daily resolution
                int(start_time.timestamp()),
                int(end_time.timestamp())
            )
            
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
            # Run in thread pool since finnhub client is synchronous
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None, self.client.forex_rates, base_currency
            )
            
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
            if test_price:
                logger.info(f"Finnhub connection test successful - EURUSD: {test_price['price']}")
                return True
            return False
        except Exception as e:
            logger.error(f"Finnhub connection test failed: {e}")
            return False