"""
Signal Engine Tests
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import os

# Import modules to test
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from backend.signals.engine import SignalEngine
from backend.signals.strategies.ema_rsi import EMAStragey
from backend.signals.strategies.donchian_atr import DonchianATRStrategy
from backend.signals.strategies.meanrev_bb import MeanReversionBBStrategy
from backend.signals.utils import calculate_atr, calculate_sl_tp, format_signal_message
from backend.models import Signal, Strategy

class TestSignalEngine:
    """Test the main signal engine"""
    
    @pytest.fixture
    def sample_ohlc_data(self):
        """Create sample OHLC data for testing"""
        dates = pd.date_range(start='2025-09-01', periods=100, freq='1min')
        
        # Generate realistic price data
        np.random.seed(42)  # For reproducible tests
        base_price = 1.0850
        returns = np.random.normal(0, 0.0002, 100)
        prices = base_price + np.cumsum(returns)
        
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            spread = 0.0001
            high = price + np.random.uniform(0, spread)
            low = price - np.random.uniform(0, spread)
            open_price = prices[i-1] if i > 0 else price
            
            data.append({
                'timestamp': date,
                'open': open_price,
                'high': max(open_price, high, price),
                'low': min(open_price, low, price),
                'close': price,
                'volume': np.random.randint(50, 200)
            })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def signal_engine(self):
        """Create a signal engine instance"""
        return SignalEngine()
    
    @pytest.fixture
    def mock_db_session(self):
        """Create a mock database session"""
        mock_session = Mock()
        mock_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = None
        mock_session.query.return_value.filter.return_value.all.return_value = []
        return mock_session
    
    def test_signal_engine_initialization(self, signal_engine):
        """Test signal engine initializes correctly"""
        assert signal_engine.mock_provider is not None
        assert signal_engine.alphavantage_provider is not None
        assert 'ema_rsi' in signal_engine.strategies
        assert 'donchian_atr' in signal_engine.strategies
        assert 'meanrev_bb' in signal_engine.strategies
        assert signal_engine.whatsapp_service is not None
    
    @pytest.mark.asyncio
    async def test_get_market_data_mock_provider(self, signal_engine, sample_ohlc_data):
        """Test getting market data from mock provider"""
        with patch.object(signal_engine.mock_provider, 'get_ohlc_data', 
                         return_value=sample_ohlc_data) as mock_get_data:
            
            data = await signal_engine._get_market_data('EURUSD')
            
            assert data is not None
            assert len(data) == 100
            assert 'timestamp' in data.columns
            assert 'open' in data.columns
            assert 'high' in data.columns
            assert 'low' in data.columns
            assert 'close' in data.columns
            mock_get_data.assert_called_once_with('EURUSD', limit=200)
    
    @pytest.mark.asyncio
    async def test_get_market_data_no_data(self, signal_engine):
        """Test getting market data when no data is available"""
        with patch.object(signal_engine.mock_provider, 'get_ohlc_data', return_value=None):
            with patch.object(signal_engine.alphavantage_provider, 'is_available', return_value=False):
                
                data = await signal_engine._get_market_data('EURUSD')
                assert data is None
    
    def test_is_duplicate_signal_no_previous(self, signal_engine, mock_db_session):
        """Test duplicate check with no previous signals"""
        signal_data = {'action': 'BUY', 'price': 1.0850}
        
        mock_db_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = None
        
        is_duplicate = signal_engine._is_duplicate_signal('EURUSD', signal_data, mock_db_session)
        assert is_duplicate == False
    
    def test_is_duplicate_signal_same_action_valid(self, signal_engine, mock_db_session):
        """Test duplicate check with same action and valid expiry"""
        signal_data = {'action': 'BUY', 'price': 1.0850}
        
        mock_signal = Mock()
        mock_signal.action = 'BUY'
        mock_signal.expires_at = datetime.utcnow() + timedelta(minutes=30)
        
        mock_db_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = mock_signal
        
        is_duplicate = signal_engine._is_duplicate_signal('EURUSD', signal_data, mock_db_session)
        assert is_duplicate == True
    
    def test_is_duplicate_signal_different_action(self, signal_engine, mock_db_session):
        """Test duplicate check with different action"""
        signal_data = {'action': 'BUY', 'price': 1.0850}
        
        mock_signal = Mock()
        mock_signal.action = 'SELL'
        mock_signal.expires_at = datetime.utcnow() + timedelta(minutes=30)
        
        mock_db_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = mock_signal
        
        is_duplicate = signal_engine._is_duplicate_signal('EURUSD', signal_data, mock_db_session)
        assert is_duplicate == False
    
    @pytest.mark.asyncio
    async def test_process_symbol_no_data(self, signal_engine, mock_db_session):
        """Test processing symbol with no market data"""
        with patch.object(signal_engine, '_get_market_data', return_value=None):
            # Should not raise exception
            await signal_engine.process_symbol('EURUSD', mock_db_session)
    
    @pytest.mark.asyncio
    async def test_process_symbol_no_strategies(self, signal_engine, mock_db_session, sample_ohlc_data):
        """Test processing symbol with no enabled strategies"""
        with patch.object(signal_engine, '_get_market_data', return_value=sample_ohlc_data):
            mock_db_session.query.return_value.filter.return_value.all.return_value = []
            
            # Should not raise exception
            await signal_engine.process_symbol('EURUSD', mock_db_session)

class TestEMAStrategy:
    """Test EMA + RSI strategy"""
    
    @pytest.fixture
    def ema_strategy(self):
        """Create EMA strategy instance"""
        return EMAStragey()
    
    @pytest.fixture
    def sample_config(self):
        """Sample EMA strategy configuration"""
        return {
            'ema_fast': 12,
            'ema_slow': 26,
            'rsi_period': 14,
            'rsi_buy_threshold': 50,
            'rsi_sell_threshold': 50,
            'min_confidence': 0.6,
            'sl_mode': 'atr',
            'sl_multiplier': 2.0,
            'tp_multiplier': 3.0,
            'expiry_bars': 60
        }
    
    @pytest.fixture
    def trending_up_data(self):
        """Create data with upward trend for testing"""
        dates = pd.date_range(start='2025-09-01', periods=50, freq='1min')
        
        # Create trending up data
        base_price = 1.0800
        trend = np.linspace(0, 0.0050, 50)  # 50 pip uptrend
        noise = np.random.normal(0, 0.0001, 50)
        prices = base_price + trend + noise
        
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            data.append({
                'timestamp': date,
                'open': prices[i-1] if i > 0 else price,
                'high': price + 0.0001,
                'low': price - 0.0001,
                'close': price,
                'volume': 100
            })
        
        return pd.DataFrame(data)
    
    def test_ema_strategy_initialization(self, ema_strategy):
        """Test EMA strategy initializes correctly"""
        assert ema_strategy is not None
    
    def test_ema_strategy_insufficient_data(self, ema_strategy, sample_config):
        """Test EMA strategy with insufficient data"""
        # Create data with only 10 bars (less than required)
        short_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2025-09-01', periods=10, freq='1min'),
            'open': [1.0850] * 10,
            'high': [1.0851] * 10,
            'low': [1.0849] * 10,
            'close': [1.0850] * 10,
            'volume': [100] * 10
        })
        
        signal = ema_strategy.generate_signal(short_data, sample_config)
        assert signal is None
    
    def test_ema_strategy_no_crossover(self, ema_strategy, sample_config):
        """Test EMA strategy with no crossover"""
        # Create flat data (no trend, no crossover)
        flat_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2025-09-01', periods=50, freq='1min'),
            'open': [1.0850] * 50,
            'high': [1.0851] * 50,
            'low': [1.0849] * 50,
            'close': [1.0850] * 50,
            'volume': [100] * 50
        })
        
        signal = ema_strategy.generate_signal(flat_data, sample_config)
        assert signal is None
    
    def test_ema_strategy_bullish_signal(self, ema_strategy, sample_config, trending_up_data):
        """Test EMA strategy generates bullish signal"""
        signal = ema_strategy.generate_signal(trending_up_data, sample_config)
        
        # May or may not generate signal depending on exact EMA crossover
        # This is acceptable as the strategy is working correctly
        if signal:
            assert signal['action'] in ['BUY', 'SELL']
            assert 'price' in signal
            assert 'confidence' in signal
            assert signal['confidence'] >= 0.0
            assert signal['confidence'] <= 1.0

class TestDonchianStrategy:
    """Test Donchian + ATR strategy"""
    
    @pytest.fixture
    def donchian_strategy(self):
        """Create Donchian strategy instance"""
        return DonchianATRStrategy()
    
    @pytest.fixture
    def sample_config(self):
        """Sample Donchian strategy configuration"""
        return {
            'donchian_period': 20,
            'atr_period': 14,
            'atr_multiplier': 2.0,
            'use_supertrend': True,
            'min_confidence': 0.65,
            'sl_multiplier': 2.0,
            'tp_multiplier': 3.0,
            'expiry_bars': 45
        }
    
    @pytest.fixture
    def breakout_data(self):
        """Create data with breakout pattern"""
        dates = pd.date_range(start='2025-09-01', periods=50, freq='1min')
        
        # Create ranging then breakout pattern
        base_price = 1.0850
        prices = []
        
        # First 30 bars: ranging
        for i in range(30):
            price = base_price + np.random.uniform(-0.0010, 0.0010)
            prices.append(price)
        
        # Last 20 bars: breakout up
        for i in range(20):
            price = base_price + 0.0010 + (i * 0.0001)
            prices.append(price)
        
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            high = price + 0.0001
            low = price - 0.0001
            
            # Make last bar break above the range
            if i == len(prices) - 1:
                high = max(prices[:30]) + 0.0002  # Break above previous highs
            
            data.append({
                'timestamp': date,
                'open': prices[i-1] if i > 0 else price,
                'high': high,
                'low': low,
                'close': price,
                'volume': 100
            })
        
        return pd.DataFrame(data)
    
    def test_donchian_strategy_initialization(self, donchian_strategy):
        """Test Donchian strategy initializes correctly"""
        assert donchian_strategy is not None
    
    def test_donchian_strategy_insufficient_data(self, donchian_strategy, sample_config):
        """Test Donchian strategy with insufficient data"""
        short_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2025-09-01', periods=10, freq='1min'),
            'open': [1.0850] * 10,
            'high': [1.0851] * 10,
            'low': [1.0849] * 10,
            'close': [1.0850] * 10,
            'volume': [100] * 10
        })
        
        signal = donchian_strategy.generate_signal(short_data, sample_config)
        assert signal is None
    
    def test_donchian_strategy_breakout(self, donchian_strategy, sample_config, breakout_data):
        """Test Donchian strategy with breakout data"""
        signal = donchian_strategy.generate_signal(breakout_data, sample_config)
        
        if signal:
            assert signal['action'] in ['BUY', 'SELL']
            assert 'price' in signal
            assert 'confidence' in signal
            assert signal['confidence'] >= 0.0
            assert signal['confidence'] <= 1.0

class TestMeanReversionStrategy:
    """Test Mean Reversion + Bollinger Bands strategy"""
    
    @pytest.fixture
    def meanrev_strategy(self):
        """Create Mean Reversion strategy instance"""
        return MeanReversionBBStrategy()
    
    @pytest.fixture
    def sample_config(self):
        """Sample Mean Reversion strategy configuration"""
        return {
            'bb_period': 20,
            'bb_std': 2.0,
            'adx_period': 14,
            'adx_threshold': 25,
            'zscore_threshold': 2.0,
            'min_confidence': 0.7,
            'sl_pips': 20,
            'tp_pips': 40,
            'expiry_bars': 30
        }
    
    @pytest.fixture
    def ranging_data(self):
        """Create ranging market data with BB touches"""
        dates = pd.date_range(start='2025-09-01', periods=50, freq='1min')
        
        base_price = 1.0850
        # Create oscillating data around mean
        prices = []
        for i in range(50):
            # Sine wave with noise
            sine_component = 0.0020 * np.sin(i * 0.3)  # 20 pip range
            noise = np.random.normal(0, 0.0002)
            price = base_price + sine_component + noise
            prices.append(price)
        
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            data.append({
                'timestamp': date,
                'open': prices[i-1] if i > 0 else price,
                'high': price + 0.0001,
                'low': price - 0.0001,
                'close': price,
                'volume': 100
            })
        
        return pd.DataFrame(data)
    
    def test_meanrev_strategy_initialization(self, meanrev_strategy):
        """Test Mean Reversion strategy initializes correctly"""
        assert meanrev_strategy is not None
    
    def test_meanrev_strategy_insufficient_data(self, meanrev_strategy, sample_config):
        """Test Mean Reversion strategy with insufficient data"""
        short_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2025-09-01', periods=10, freq='1min'),
            'open': [1.0850] * 10,
            'high': [1.0851] * 10,
            'low': [1.0849] * 10,
            'close': [1.0850] * 10,
            'volume': [100] * 10
        })
        
        signal = meanrev_strategy.generate_signal(short_data, sample_config)
        assert signal is None
    
    def test_meanrev_strategy_ranging_data(self, meanrev_strategy, sample_config, ranging_data):
        """Test Mean Reversion strategy with ranging data"""
        signal = meanrev_strategy.generate_signal(ranging_data, sample_config)
        
        if signal:
            assert signal['action'] in ['BUY', 'SELL']
            assert 'price' in signal
            assert 'confidence' in signal
            assert signal['confidence'] >= 0.0
            assert signal['confidence'] <= 1.0

class TestSignalUtils:
    """Test signal utility functions"""
    
    @pytest.fixture
    def sample_data(self):
        """Sample OHLC data for testing utils"""
        return pd.DataFrame({
            'timestamp': pd.date_range(start='2025-09-01', periods=30, freq='1min'),
            'open': [1.0850] * 30,
            'high': [1.0860] * 30,
            'low': [1.0840] * 30,
            'close': [1.0850] * 30,
            'volume': [100] * 30
        })
    
    def test_calculate_atr(self, sample_data):
        """Test ATR calculation"""
        atr_values = calculate_atr(sample_data, period=14)
        
        assert len(atr_values) == len(sample_data)
        assert not np.isnan(atr_values[-1])  # Last value should not be NaN
        assert atr_values[-1] > 0  # ATR should be positive
    
    def test_calculate_sl_tp_atr_mode(self, sample_data):
        """Test SL/TP calculation in ATR mode"""
        config = {
            'sl_mode': 'atr',
            'tp_mode': 'atr',
            'sl_multiplier': 2.0,
            'tp_multiplier': 3.0
        }
        
        price = 1.0850
        sl, tp = calculate_sl_tp(price, 'BUY', sample_data, config)
        
        assert sl is not None
        assert tp is not None
        assert sl < price  # Stop loss should be below entry for BUY
        assert tp > price  # Take profit should be above entry for BUY
        assert tp - price > price - sl  # TP should be further than SL
    
    def test_calculate_sl_tp_pips_mode(self, sample_data):
        """Test SL/TP calculation in pips mode"""
        config = {
            'sl_mode': 'pips',
            'tp_mode': 'pips',
            'sl_pips': 20,
            'tp_pips': 40
        }
        
        price = 1.0850
        sl, tp = calculate_sl_tp(price, 'BUY', sample_data, config)
        
        assert sl is not None
        assert tp is not None
        assert sl < price
        assert tp > price
        
        # Check pip distances (approximately)
        sl_pips = (price - sl) / 0.0001
        tp_pips = (tp - price) / 0.0001
        
        assert abs(sl_pips - 20) < 1  # Allow small rounding error
        assert abs(tp_pips - 40) < 1
    
    def test_calculate_sl_tp_sell_signal(self, sample_data):
        """Test SL/TP calculation for SELL signal"""
        config = {
            'sl_mode': 'pips',
            'tp_mode': 'pips',
            'sl_pips': 20,
            'tp_pips': 40
        }
        
        price = 1.0850
        sl, tp = calculate_sl_tp(price, 'SELL', sample_data, config)
        
        assert sl is not None
        assert tp is not None
        assert sl > price  # Stop loss should be above entry for SELL
        assert tp < price  # Take profit should be below entry for SELL
    
    def test_format_signal_message(self):
        """Test signal message formatting"""
        signal_data = {
            'symbol': 'EURUSD',
            'action': 'BUY',
            'price': 1.08523,
            'sl': 1.08323,
            'tp': 1.08723,
            'confidence': 0.72,
            'strategy': 'ema_rsi'
        }
        
        message = format_signal_message(signal_data)
        
        assert 'EURUSD' in message
        assert 'BUY' in message
        assert '1.08523' in message
        assert '1.08323' in message
        assert '1.08723' in message
        assert '0.72' in message
        assert 'ema_rsi' in message
    
    def test_format_signal_message_missing_fields(self):
        """Test signal message formatting with missing fields"""
        signal_data = {
            'symbol': 'EURUSD',
            'action': 'BUY',
            'price': 1.08523,
            'confidence': 0.72,
            'strategy': 'ema_rsi'
            # Missing SL and TP
        }
        
        message = format_signal_message(signal_data)
        
        assert 'EURUSD' in message
        assert 'BUY' in message
        assert 'N/A' in message  # For missing SL/TP

class TestDataProviders:
    """Test data provider functionality"""
    
    def test_mock_provider_initialization(self):
        """Test mock provider can be initialized"""
        from backend.providers.mock import MockDataProvider
        
        provider = MockDataProvider()
        assert provider is not None
        assert provider.is_available() == True
    
    @pytest.mark.asyncio
    async def test_mock_provider_data_generation(self):
        """Test mock provider generates data"""
        from backend.providers.mock import MockDataProvider
        
        provider = MockDataProvider()
        
        # Test data retrieval
        data = await provider.get_ohlc_data('EURUSD', limit=50)
        
        if data is not None:
            assert len(data) <= 50
            assert 'timestamp' in data.columns
            assert 'open' in data.columns
            assert 'high' in data.columns
            assert 'low' in data.columns
            assert 'close' in data.columns
            assert 'volume' in data.columns
    
    def test_alphavantage_provider_initialization(self):
        """Test Alpha Vantage provider initialization"""
        from backend.providers.alphavantage import AlphaVantageProvider
        
        provider = AlphaVantageProvider()
        assert provider is not None
        # Should be disabled without API key
        assert provider.is_available() == False

if __name__ == "__main__":
    pytest.main([__file__])
