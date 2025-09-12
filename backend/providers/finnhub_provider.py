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
        """Get synthetic OHLC data from current Finnhub prices"""
        if not self.is_available():
            return None
            
        try:
            # Get current price from Finnhub (this is free)
            current_price_data = await self.get_current_price(symbol)
            if not current_price_data:
                logger.warning(f"No current price data from Finnhub for {symbol}")
                return None
            
            # Create synthetic OHLC data based on current price
            # This simulates realistic forex price movements
            current_price = current_price_data['price']
            high_price = current_price_data['high'] or current_price * 1.001
            low_price = current_price_data['low'] or current_price * 0.999
            open_price = current_price_data['previous_close'] or current_price
            
            # Generate synthetic historical data with realistic variations
            dates = pd.date_range(end=datetime.now(), periods=limit, freq='1min')
            
            # Create price variations around current price (±0.5%)
            np_random = __import__('numpy').random
            price_variations = np_random.normal(0, 0.002, limit)  # 0.2% standard deviation
            
            base_prices = [current_price * (1 + var) for var in price_variations]
            
            df_data = []
            for i, date in enumerate(dates):
                base = base_prices[i]
                daily_var = np_random.normal(0, 0.001)  # Daily variation
                
                df_data.append({
                    'time': date,
                    'open': base * (1 + daily_var),
                    'high': base * (1 + abs(daily_var) + 0.0005),
                    'low': base * (1 - abs(daily_var) - 0.0005),
                    'close': base,
                    'volume': np_random.randint(1000, 10000)
                })
            
            df = pd.DataFrame(df_data)
            df = df.set_index('time').sort_index()
            
            logger.info(f"Generated {len(df)} synthetic bars for {symbol} based on Finnhub current price: {current_price}")
            return df
                        
        except Exception as e:
            logger.error(f"Finnhub synthetic OHLC error for {symbol}: {e}")
            
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