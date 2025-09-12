"""
Signal Generation Engine
"""
import pandas as pd
import os
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from typing import Optional, List

from ..models import Signal, Strategy
from ..providers.mock import MockDataProvider
from ..providers.alphavantage import AlphaVantageProvider
from ..providers.freecurrency import FreeCurrencyAPIProvider
from ..providers.mt5_data import MT5DataProvider
from ..providers.finnhub_provider import FinnhubProvider
from ..providers.exchangerate_provider import ExchangeRateProvider
from ..risk.guards import RiskManager
from ..services.whatsapp import WhatsAppService
from ..regime.detector import regime_detector
from ..providers.execution.mt5_bridge import MT5BridgeExecutionProvider
from ..providers.execution.base import OrderRequest, OrderType
from ..logs.logger import get_logger
from .strategies.ema_rsi import EMAStragey
from .strategies.donchian_atr import DonchianATRStrategy
from .strategies.meanrev_bb import MeanReversionBBStrategy
from .strategies.macd_strategy import MACDStrategy
from .strategies.stochastic_strategy import StochasticStrategy
from .strategies.rsi_divergence import RSIDivergenceStrategy
from .strategies.fibonacci_strategy import FibonacciStrategy

logger = get_logger(__name__)

class SignalEngine:
    """Main signal generation engine"""
    
    def __init__(self):
        # Initialize data providers (priority order: ExchangeRate.host -> Finnhub -> FreeCurrency -> Alpha Vantage -> Mock)
        self.exchangerate_provider = ExchangeRateProvider()
        self.finnhub_provider = FinnhubProvider()
        self.mt5_provider = MT5DataProvider()
        self.freecurrency_provider = FreeCurrencyAPIProvider()
        self.alphavantage_provider = AlphaVantageProvider()
        self.mock_provider = MockDataProvider()
        
        # Strategy mapping - 7 Advanced Trading Strategies
        self.strategies = {
            'ema_rsi': EMAStragey(),
            'donchian_atr': DonchianATRStrategy(),
            'meanrev_bb': MeanReversionBBStrategy(),
            'macd_crossover': MACDStrategy(),
            'stochastic': StochasticStrategy(),
            'rsi_divergence': RSIDivergenceStrategy(),
            'fibonacci': FibonacciStrategy()
        }
        
        self.whatsapp_service = WhatsAppService()
        self.execution_provider = MT5BridgeExecutionProvider()
        
        # Auto-trading configuration
        self.auto_trade_enabled = os.getenv('AUTO_TRADE_ENABLED', 'false').lower() == 'true'
        self.confidence_threshold = float(os.getenv('AUTO_TRADE_CONFIDENCE_THRESHOLD', '0.85'))
        self.default_lot_size = float(os.getenv('AUTO_TRADE_LOT_SIZE', '0.01'))  # Micro lot
    
    async def process_symbol(self, symbol: str, db: Session):
        """Process signals for a single symbol"""
        try:
            logger.debug(f"Processing signals for {symbol}")
            
            # Get initial OHLC data
            data = await self._get_market_data(symbol)
            if data is None:
                logger.warning(f"No data available for {symbol}")
                return
            
            # Add a new bar to simulate real-time updates
            if hasattr(self.mock_provider, 'add_new_bar'):
                self.mock_provider.add_new_bar(symbol)
                # Refresh data with new bar
                data = await self._get_market_data(symbol)
                
            # Check data sufficiency after refresh
            if data is None or len(data) < 30:
                logger.warning(f"Insufficient data for {symbol}: {len(data) if data is not None else 0} bars")
                return
            
            # Get strategy configurations for this symbol
            strategies = db.query(Strategy).filter(
                Strategy.symbol == symbol,
                Strategy.enabled == True
            ).all()
            
            if not strategies:
                logger.debug(f"No enabled strategies for {symbol}")
                return
            
            # Detect market regime first
            regime_data = regime_detector.detect_regime(data, symbol)
            if regime_data['regime'] != 'UNKNOWN':
                regime_detector.store_regime(symbol, regime_data, db)
                logger.debug(f"Market regime for {symbol}: {regime_data['regime']} ({regime_data['confidence']:.2f})")
            
            # Process each strategy
            for strategy_config in strategies:
                # Check if strategy is suitable for current regime
                if not regime_detector.is_regime_suitable_for_strategy(regime_data['regime'], strategy_config.name):
                    logger.debug(f"Strategy {strategy_config.name} skipped for {symbol} - unsuitable for {regime_data['regime']} regime")
                    continue
                    
                await self._process_strategy(symbol, data, strategy_config, db)
                
        except Exception as e:
            logger.error(f"Error processing symbol {symbol}: {e}")
    
    async def _get_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get market data from available providers (priority: ExchangeRate.host -> Finnhub -> FreeCurrency -> Alpha Vantage -> Mock)"""
        # Try ExchangeRate.host first for free unlimited forex data
        if self.exchangerate_provider.is_available():
            data = await self.exchangerate_provider.get_ohlc_data(symbol, limit=200)
            if data is not None:
                logger.info(f"Using ExchangeRate.host live forex data for {symbol}")
                return data
        
        # Try Finnhub for real-time data (if available)
        if self.finnhub_provider.is_available():
            data = await self.finnhub_provider.get_ohlc_data(symbol, limit=200)
            if data is not None:
                logger.info(f"Using Finnhub real-time forex data for {symbol}")
                return data
        
        # Try FreeCurrencyAPI for live data
        if self.freecurrency_provider.is_available():
            data = await self.freecurrency_provider.get_ohlc_data(symbol, limit=200)
            if data is not None:
                logger.debug(f"Using FreeCurrency live data for {symbol}")
                return data
        
        # Try Alpha Vantage if available
        if self.alphavantage_provider.is_available():
            data = await self.alphavantage_provider.get_ohlc_data(symbol, limit=200)
            if data is not None:
                logger.debug(f"Using Alpha Vantage data for {symbol}")
                return data
        
        # Fallback to mock data
        data = await self.mock_provider.get_ohlc_data(symbol, limit=200)
        if data is not None:
            logger.debug(f"Using mock data for {symbol}")
            return data
        
        logger.error(f"No data available for {symbol}")
        return None
    
    async def _process_strategy(
        self, 
        symbol: str, 
        data: pd.DataFrame, 
        strategy_config: Strategy, 
        db: Session
    ):
        """Process a single strategy for a symbol"""
        try:
            strategy_name = strategy_config.name
            if strategy_name not in self.strategies:
                logger.error(f"Unknown strategy: {strategy_name}")
                return
            
            strategy = self.strategies[strategy_name]
            
            # Generate signal
            signal_data = strategy.generate_signal(data, strategy_config.config)
            
            if signal_data is None:
                logger.debug(f"No signal generated for {symbol} using {strategy_name}")
                return
            
            # Check for signal deduplication and conflicts
            if self._is_duplicate_signal(symbol, signal_data, strategy_name, db):
                logger.debug(f"Duplicate/conflicting signal filtered for {symbol} using {strategy_name}")
                return
                
            # Check cross-strategy consensus
            if not self._get_cross_strategy_consensus(symbol, signal_data, db):
                logger.debug(f"Cross-strategy consensus check failed for {symbol} using {strategy_name}")
                return
            
            # Create signal object
            expires_at = datetime.utcnow() + timedelta(
                minutes=strategy_config.config.get('expiry_bars', 60)
            )
            
            # Convert numpy types to Python types for database compatibility
            price = float(signal_data['price']) if signal_data['price'] is not None else None
            sl = float(signal_data.get('sl')) if signal_data.get('sl') is not None else None
            tp = float(signal_data.get('tp')) if signal_data.get('tp') is not None else None
            confidence = float(signal_data['confidence']) if signal_data['confidence'] is not None else None
            
            signal = Signal(
                symbol=symbol,
                action=signal_data['action'],
                price=price,
                sl=sl,
                tp=tp,
                confidence=confidence,
                strategy=strategy_name,
                expires_at=expires_at
            )
            
            # Apply risk management
            risk_manager = RiskManager(db)
            risk_check = risk_manager.check_signal(signal, data)
            
            if not risk_check['allowed']:
                signal.blocked_by_risk = True
                signal.risk_reason = risk_check['reason']
                logger.info(f"Signal blocked by risk management: {risk_check['reason']}")
            else:
                # Send to WhatsApp if not blocked
                try:
                    await self.whatsapp_service.send_signal(signal)
                    signal.sent_to_whatsapp = True
                    logger.info(f"Signal sent to WhatsApp for {symbol}")
                except Exception as e:
                    logger.error(f"Failed to send WhatsApp message: {e}")
                
                # Automatic trade execution if enabled and confidence threshold met
                if self.auto_trade_enabled and signal.confidence >= self.confidence_threshold:
                    try:
                        await self._execute_auto_trade(signal, db)
                    except Exception as e:
                        logger.error(f"Auto-trade execution failed for signal {signal.id}: {e}")
            
            # Save signal to database
            db.add(signal)
            db.commit()
            
            logger.info(f"Signal created for {symbol}: {signal_data['action']} @ {signal_data['price']}")
            
        except Exception as e:
            logger.error(f"Error processing strategy {strategy_config.name} for {symbol}: {e}")
            db.rollback()
    
    async def _execute_auto_trade(self, signal: Signal, db: Session):
        """Execute automatic trade for high-confidence signals"""
        try:
            logger.info(f"Executing auto-trade for signal {signal.id}: {signal.action} {signal.symbol} @ {signal.price} (confidence: {signal.confidence:.1%})")
            
            # Determine order type
            if signal.action == 'BUY':
                order_type = OrderType.MARKET_BUY
            elif signal.action == 'SELL':
                order_type = OrderType.MARKET_SELL
            else:
                raise ValueError(f"Unknown signal action: {signal.action}")
            
            # Create order request
            order_request = OrderRequest(
                symbol=signal.symbol,
                order_type=order_type,
                volume=self.default_lot_size,
                price=None,  # Market order
                stop_loss=signal.sl,
                take_profit=signal.tp,
                comment=f"AutoTrade-{signal.id}-{signal.strategy}",
                magic_number=234000 + signal.id  # Unique magic number
            )
            
            # Execute order through MT5 bridge
            execution_result = await self.execution_provider.execute_order(order_request)
            
            if execution_result.success:
                # Update signal with execution details
                signal.auto_traded = True
                signal.broker_ticket = execution_result.ticket
                signal.executed_price = execution_result.executed_price
                signal.executed_volume = execution_result.executed_volume
                signal.execution_slippage = execution_result.slippage
                signal.execution_time = execution_result.execution_time
                
                logger.info(f"Auto-trade executed successfully - Ticket: {execution_result.ticket}, Price: {execution_result.executed_price}")
                
            else:
                # Log execution failure
                signal.auto_trade_failed = True
                signal.execution_error = execution_result.message
                
                logger.error(f"Auto-trade execution failed: {execution_result.message}")
                
        except Exception as e:
            # Update signal with failure info
            signal.auto_trade_failed = True
            signal.execution_error = str(e)
            
            logger.error(f"Auto-trade execution error: {e}")
            raise
    
    def _is_duplicate_signal(self, symbol: str, signal_data: dict, strategy_name: str, db: Session) -> bool:
        """Check if this signal conflicts with recent signals (enhanced logic)"""
        try:
            now = datetime.utcnow()
            cooldown_minutes = 15  # No new signals for same symbol within 15 minutes
            
            # Check for any recent signal for this symbol (regardless of strategy)
            recent_signal = db.query(Signal).filter(
                Signal.symbol == symbol,
                Signal.issued_at > now - timedelta(minutes=cooldown_minutes)
            ).order_by(Signal.issued_at.desc()).first()
            
            if recent_signal:
                # If recent signal has same action, it's a duplicate
                if recent_signal.action == signal_data['action']:
                    logger.debug(f"Duplicate signal blocked for {symbol}: same action within cooldown")
                    return True
                    
                # If recent signal has opposite action, check confidence levels
                if recent_signal.action != signal_data['action']:
                    # Only allow opposite signal if new signal has significantly higher confidence
                    confidence_threshold = recent_signal.confidence + 0.15  # 15% higher confidence required
                    if signal_data['confidence'] < confidence_threshold:
                        logger.debug(f"Conflicting signal blocked for {symbol}: insufficient confidence {signal_data['confidence']:.2f} vs required {confidence_threshold:.2f}")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking duplicate signal: {e}")
            return False
            
    def _get_cross_strategy_consensus(self, symbol: str, signal_data: dict, db: Session) -> bool:
        """Check if multiple strategies agree on signal direction"""
        try:
            now = datetime.utcnow()
            recent_minutes = 10  # Look at signals in last 10 minutes
            
            # Get recent signals for this symbol from all strategies
            recent_signals = db.query(Signal).filter(
                Signal.symbol == symbol,
                Signal.issued_at > now - timedelta(minutes=recent_minutes),
                Signal.blocked_by_risk == False
            ).all()
            
            if not recent_signals:
                return True  # No recent signals, allow
                
            # Count signals by action
            buy_count = sum(1 for s in recent_signals if s.action == 'BUY')
            sell_count = sum(1 for s in recent_signals if s.action == 'SELL')
            
            # Current signal action
            current_action = signal_data['action']
            
            # If there are conflicting signals, require higher confidence
            if (current_action == 'BUY' and sell_count > 0) or (current_action == 'SELL' and buy_count > 0):
                # Require 80%+ confidence for conflicting signals
                if signal_data['confidence'] < 0.80:
                    logger.debug(f"Cross-strategy conflict: {symbol} needs 80%+ confidence for {current_action}, got {signal_data['confidence']:.2f}")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error checking cross-strategy consensus: {e}")
            return True  # Allow on error
