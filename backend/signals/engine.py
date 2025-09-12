"""
Signal Generation Engine
"""
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from typing import Optional, List

from ..models import Signal, Strategy
from ..providers.mock import MockDataProvider
from ..providers.alphavantage import AlphaVantageProvider
from ..risk.guards import RiskManager
from ..services.whatsapp import WhatsAppService
from ..logs.logger import get_logger
from .strategies.ema_rsi import EMAStragey
from .strategies.donchian_atr import DonchianATRStrategy
from .strategies.meanrev_bb import MeanReversionBBStrategy

logger = get_logger(__name__)

class SignalEngine:
    """Main signal generation engine"""
    
    def __init__(self):
        # Initialize data providers
        self.mock_provider = MockDataProvider()
        self.alphavantage_provider = AlphaVantageProvider()
        
        # Strategy mapping
        self.strategies = {
            'ema_rsi': EMAStragey(),
            'donchian_atr': DonchianATRStrategy(),
            'meanrev_bb': MeanReversionBBStrategy()
        }
        
        self.whatsapp_service = WhatsAppService()
    
    async def process_symbol(self, symbol: str, db: Session):
        """Process signals for a single symbol"""
        try:
            logger.debug(f"Processing signals for {symbol}")
            
            # Get OHLC data
            data = await self._get_market_data(symbol)
            if data is None or len(data) < 50:
                logger.warning(f"Insufficient data for {symbol}")
                return
            
            # Add a new bar to simulate real-time updates
            if hasattr(self.mock_provider, 'add_new_bar'):
                self.mock_provider.add_new_bar(symbol)
                # Refresh data with new bar
                data = await self._get_market_data(symbol)
            
            # Get strategy configurations for this symbol
            strategies = db.query(Strategy).filter(
                Strategy.symbol == symbol,
                Strategy.enabled == True
            ).all()
            
            if not strategies:
                logger.debug(f"No enabled strategies for {symbol}")
                return
            
            # Process each strategy
            for strategy_config in strategies:
                await self._process_strategy(symbol, data, strategy_config, db)
                
        except Exception as e:
            logger.error(f"Error processing symbol {symbol}: {e}")
    
    async def _get_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get market data from available providers"""
        # Try Alpha Vantage first if available
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
            
            # Check for signal deduplication
            if self._is_duplicate_signal(symbol, signal_data, db):
                logger.debug(f"Duplicate signal filtered for {symbol}")
                return
            
            # Create signal object
            expires_at = datetime.utcnow() + timedelta(
                minutes=strategy_config.config.get('expiry_bars', 60)
            )
            
            signal = Signal(
                symbol=symbol,
                action=signal_data['action'],
                price=signal_data['price'],
                sl=signal_data.get('sl'),
                tp=signal_data.get('tp'),
                confidence=signal_data['confidence'],
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
            
            # Save signal to database
            db.add(signal)
            db.commit()
            
            logger.info(f"Signal created for {symbol}: {signal_data['action']} @ {signal_data['price']}")
            
        except Exception as e:
            logger.error(f"Error processing strategy {strategy_config.name} for {symbol}: {e}")
            db.rollback()
    
    def _is_duplicate_signal(self, symbol: str, signal_data: dict, db: Session) -> bool:
        """Check if this signal is a duplicate of the last one"""
        try:
            # Get the last signal for this symbol
            last_signal = db.query(Signal).filter(
                Signal.symbol == symbol
            ).order_by(Signal.issued_at.desc()).first()
            
            if not last_signal:
                return False
            
            # Check if action is the same and signal is still valid
            if (last_signal.action == signal_data['action'] and 
                last_signal.expires_at > datetime.utcnow()):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking duplicate signal: {e}")
            return False
