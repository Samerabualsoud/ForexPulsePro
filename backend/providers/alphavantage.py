"""
Alpha Vantage Data Provider (Stub Implementation)
"""
import os
import requests
import pandas as pd
from typing import Optional
from datetime import datetime

from .base import BaseDataProvider
from ..logs.logger import get_logger

logger = get_logger(__name__)

class AlphaVantageProvider(BaseDataProvider):
    """Alpha Vantage API data provider"""
    
    def __init__(self):
        self.api_key = os.getenv("ALPHAVANTAGE_KEY")
        self.base_url = "https://www.alphavantage.co/query"
        self.enabled = bool(self.api_key)
        
        if not self.enabled:
            logger.info("Alpha Vantage provider disabled - no API key provided")
    
    async def get_ohlc_data(
        self, 
        symbol: str, 
        timeframe: str = "M1", 
        limit: int = 100
    ) -> Optional[pd.DataFrame]:
        """
        Get OHLC data from Alpha Vantage API
        Note: This is a stub implementation - Alpha Vantage has limited forex support
        """
        if not self.enabled:
            return None
        
        try:
            # Convert symbol format (EURUSD -> EUR/USD)
            if len(symbol) == 6:
                from_currency = symbol[:3]
                to_currency = symbol[3:]
            else:
                logger.error(f"Invalid symbol format: {symbol}")
                return None
            
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
                "outputsize": "compact"  # Last 100 data points
            }
            
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if "Error Message" in data:
                logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                return None
            
            if "Note" in data:
                logger.warning(f"Alpha Vantage API limit: {data['Note']}")
                return None
            
            # Parse time series data
            time_series_key = f"Time Series FX ({interval})"
            if time_series_key not in data:
                logger.error(f"No time series data found for {symbol}")
                return None
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            rows = []
            for timestamp_str, ohlc in time_series.items():
                rows.append({
                    'timestamp': pd.to_datetime(timestamp_str),
                    'open': float(ohlc['1. open']),
                    'high': float(ohlc['2. high']),
                    'low': float(ohlc['3. low']),
                    'close': float(ohlc['4. close']),
                    'volume': 0  # Forex doesn't have volume in Alpha Vantage
                })
            
            df = pd.DataFrame(rows)
            df = df.sort_values('timestamp')
            
            if limit:
                df = df.tail(limit)
            
            logger.info(f"Retrieved {len(df)} bars for {symbol} from Alpha Vantage")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error retrieving data for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving data for {symbol}: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if Alpha Vantage is configured and available"""
        return self.enabled
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price from Alpha Vantage"""
        if not self.enabled:
            return None
        
        try:
            # Convert symbol format
            if len(symbol) == 6:
                from_currency = symbol[:3]
                to_currency = symbol[3:]
            else:
                return None
            
            params = {
                "function": "CURRENCY_EXCHANGE_RATE",
                "from_currency": from_currency,
                "to_currency": to_currency,
                "apikey": self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if "Realtime Currency Exchange Rate" in data:
                rate_data = data["Realtime Currency Exchange Rate"]
                return float(rate_data["5. Exchange Rate"])
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {e}")
            return None
