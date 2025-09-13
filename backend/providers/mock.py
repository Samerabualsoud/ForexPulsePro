"""
Mock Data Provider - Uses CSV files or generates synthetic data
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path
from typing import Optional

from .base import BaseDataProvider
from ..logs.logger import get_logger

logger = get_logger(__name__)

class MockDataProvider(BaseDataProvider):
    """Mock data provider using CSV files or synthetic data generation"""
    
    def __init__(self):
        self.data_dir = Path("data/mock")
        self.data_cache = {}
        self._ensure_mock_data()
    
    def _ensure_mock_data(self):
        """Ensure mock data files exist, generate if missing"""
        symbols = [
            'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 
            'USDCHF', 'NZDUSD', 'EURGBP', 'EURJPY', 'GBPJPY',
            'AUDJPY', 'CHFJPY', 'EURCHF', 'GBPAUD', 'AUDCAD',
            'BTCUSD'  # Bitcoin support added
        ]
        
        for symbol in symbols:
            csv_path = self.data_dir / f"{symbol}.csv"
            if not csv_path.exists():
                logger.info(f"Generating synthetic data for {symbol}")
                self._generate_synthetic_data(symbol, csv_path)
    
    def _generate_synthetic_data(self, symbol: str, filepath: Path):
        """Generate synthetic OHLC data"""
        # Base prices for different pairs (realistic 2025 levels)
        base_prices = {
            'EURUSD': 1.0850, 'GBPUSD': 1.2650, 'USDJPY': 149.50,
            'AUDUSD': 0.6420, 'USDCAD': 1.4350, 'USDCHF': 0.8950,
            'NZDUSD': 0.5680, 'EURGBP': 0.8580, 'EURJPY': 162.30,
            'GBPJPY': 189.20, 'AUDJPY': 96.00, 'CHFJPY': 167.10,
            'EURCHF': 0.9710, 'GBPAUD': 1.9700, 'AUDCAD': 0.9210,
            'BTCUSD': 42000.00  # Bitcoin realistic price level
        }
        
        base_price = base_prices.get(symbol, 1.0000)
        
        # Generate 7 days of minute data
        start_time = datetime.utcnow() - timedelta(days=7)
        periods = 7 * 24 * 60  # 7 days * 24 hours * 60 minutes
        
        # Create time series
        timestamps = pd.date_range(start=start_time, periods=periods, freq='1min')
        
        # Generate realistic price movements
        np.random.seed(42)  # For reproducible data
        returns = np.random.normal(0, 0.0002, periods)  # Small random movements
        
        # Add some trending and volatility clustering
        trend = np.sin(np.arange(periods) / 1440 * 2 * np.pi) * 0.001  # Daily trend
        volatility_mult = 1 + 0.5 * np.sin(np.arange(periods) / 360 * 2 * np.pi)
        
        price_changes = (returns + trend) * volatility_mult
        prices = base_price + np.cumsum(price_changes)
        
        # Generate OHLC from prices
        data = []
        for i, (timestamp, price) in enumerate(zip(timestamps, prices)):
            # Random intrabar movement
            spread = np.random.uniform(0.0001, 0.0003)
            high = price + np.random.uniform(0, spread)
            low = price - np.random.uniform(0, spread)
            
            # Open is previous close (with small gap)
            if i == 0:
                open_price = price
            else:
                open_price = data[i-1]['close'] + np.random.uniform(-0.0001, 0.0001)
            
            close_price = price
            volume = np.random.randint(50, 200)
            
            data.append({
                'timestamp': timestamp,
                'open': round(open_price, 5),
                'high': round(max(open_price, high, close_price), 5),
                'low': round(min(open_price, low, close_price), 5),
                'close': round(close_price, 5),
                'volume': volume
            })
        
        # Save to CSV
        df = pd.DataFrame(data)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f"Generated {len(df)} data points for {symbol}")
    
    async def get_ohlc_data(
        self, 
        symbol: str, 
        timeframe: str = "M1", 
        limit: int = 100
    ) -> Optional[pd.DataFrame]:
        """Get OHLC data from CSV file"""
        try:
            csv_path = self.data_dir / f"{symbol}.csv"
            
            if not csv_path.exists():
                logger.warning(f"No data file found for {symbol}")
                return None
            
            # Read CSV
            df = pd.read_csv(csv_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Return latest N bars
            if limit:
                df = df.tail(limit)
            
            # Update cache
            self.data_cache[symbol] = df
            
            return df
            
        except Exception as e:
            logger.error(f"Error reading data for {symbol}: {e}")
            return None
    
    def is_available(self) -> bool:
        """Mock provider is always available"""
        return True
    
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price from cached data"""
        try:
            if symbol in self.data_cache:
                return float(self.data_cache[symbol]['close'].iloc[-1])
            
            # Try to load from file
            csv_path = self.data_dir / f"{symbol}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                return float(df['close'].iloc[-1])
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {e}")
            return None
    
    def add_new_bar(self, symbol: str):
        """Add a new bar to simulate real-time data"""
        try:
            csv_path = self.data_dir / f"{symbol}.csv"
            if not csv_path.exists():
                return
            
            df = pd.read_csv(csv_path)
            last_row = df.iloc[-1].copy()
            
            # Generate new bar
            last_close = last_row['close']
            change = np.random.normal(0, 0.0002)
            
            new_timestamp = pd.to_datetime(last_row['timestamp']) + timedelta(minutes=1)
            new_open = last_close + np.random.uniform(-0.0001, 0.0001)
            new_close = new_open + change
            
            spread = np.random.uniform(0.0001, 0.0003)
            new_high = max(new_open, new_close) + np.random.uniform(0, spread/2)
            new_low = min(new_open, new_close) - np.random.uniform(0, spread/2)
            
            new_row = {
                'timestamp': new_timestamp,
                'open': round(new_open, 5),
                'high': round(new_high, 5),
                'low': round(new_low, 5),
                'close': round(new_close, 5),
                'volume': np.random.randint(50, 200)
            }
            
            # Append to file
            new_df = pd.DataFrame([new_row])
            new_df.to_csv(csv_path, mode='a', header=False, index=False)
            
            logger.debug(f"Added new bar for {symbol}: {new_close}")
            
        except Exception as e:
            logger.error(f"Error adding new bar for {symbol}: {e}")
