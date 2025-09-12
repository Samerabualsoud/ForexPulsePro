"""
Risk Management Guards
"""
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from typing import Dict, Any, List

from ..models import RiskConfig, Signal
from ..signals.utils import calculate_atr
from ..logs.logger import get_logger

logger = get_logger(__name__)

class RiskManager:
    """Risk management system with multiple safety guards"""
    
    def __init__(self, db: Session):
        self.db = db
        self.config = self._get_risk_config()
    
    def _get_risk_config(self) -> RiskConfig:
        """Get current risk configuration"""
        config = self.db.query(RiskConfig).first()
        if not config:
            # Create default config
            config = RiskConfig()
            self.db.add(config)
            self.db.commit()
        return config
    
    def check_signal(self, signal, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check if signal should be allowed through risk filters
        
        Returns:
            Dict with 'allowed' (bool) and 'reason' (str) keys
        """
        try:
            # Refresh config
            self.config = self._get_risk_config()
            
            # Check kill switch
            if self.config.kill_switch_enabled:
                return {
                    'allowed': False,
                    'reason': 'Global kill switch is enabled'
                }
            
            # Check daily loss limit
            if not self._check_daily_loss_limit():
                return {
                    'allowed': False,
                    'reason': 'Daily loss limit exceeded'
                }
            
            # Check volatility guard
            if not self._check_volatility_guard(signal.symbol, market_data):
                return {
                    'allowed': False,
                    'reason': f'High volatility detected (ATR > {self.config.volatility_threshold*100}%)'
                }
            
            # Check daily signal limit
            if not self._check_daily_signal_limit():
                return {
                    'allowed': False,
                    'reason': 'Maximum daily signals reached'
                }
            
            # Check signal quality
            if not self._check_signal_quality(signal):
                return {
                    'allowed': False,
                    'reason': 'Signal quality below minimum threshold'
                }
            
            return {'allowed': True, 'reason': 'All risk checks passed'}
            
        except Exception as e:
            logger.error(f"Error in risk check: {e}")
            return {
                'allowed': False,
                'reason': 'Risk check system error'
            }
    
    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit has been exceeded"""
        try:
            today = datetime.utcnow().date()
            
            # Get today's signals that hit stop loss (simplified - in real system would need trade tracking)
            today_signals = self.db.query(Signal).filter(
                Signal.issued_at >= datetime.combine(today, datetime.min.time()),
                Signal.blocked_by_risk == False
            ).all()
            
            # Simplified loss calculation (in real system would track actual P&L)
            estimated_loss = len([s for s in today_signals if s.action in ['BUY', 'SELL']]) * 50  # Assume $50 avg loss per signal
            
            return estimated_loss < self.config.daily_loss_limit
            
        except Exception as e:
            logger.error(f"Error checking daily loss limit: {e}")
            return True  # Allow by default on error
    
    def _check_volatility_guard(self, symbol: str, market_data: pd.DataFrame) -> bool:
        """Check if volatility is within acceptable limits"""
        try:
            if not self.config.volatility_guard_enabled:
                return True
            
            # Calculate current ATR as percentage of price
            atr_values = calculate_atr(market_data, period=14)
            if len(atr_values) == 0 or pd.isna(atr_values[-1]):
                return True  # Allow if can't calculate
            
            current_price = market_data['close'].iloc[-1]
            atr_percentage = atr_values[-1] / current_price
            
            logger.debug(f"Volatility check for {symbol}: ATR% = {atr_percentage:.4f}, threshold = {self.config.volatility_threshold}")
            
            return atr_percentage <= self.config.volatility_threshold
            
        except Exception as e:
            logger.error(f"Error checking volatility guard: {e}")
            return True  # Allow by default on error
    
    def _check_daily_signal_limit(self) -> bool:
        """Check if maximum daily signals have been reached"""
        try:
            today = datetime.utcnow().date()
            
            today_signal_count = self.db.query(Signal).filter(
                Signal.issued_at >= datetime.combine(today, datetime.min.time()),
                Signal.blocked_by_risk == False
            ).count()
            
            return today_signal_count < self.config.max_daily_signals
            
        except Exception as e:
            logger.error(f"Error checking daily signal limit: {e}")
            return True  # Allow by default on error
    
    def _check_signal_quality(self, signal) -> bool:
        """Check signal quality metrics"""
        try:
            # Minimum confidence check
            if signal.confidence < 0.5:
                return False
            
            # Check that SL and TP are reasonable
            if signal.sl and signal.tp:
                sl_distance = abs(signal.price - signal.sl)
                tp_distance = abs(signal.price - signal.tp)
                
                # Risk/reward ratio should be at least 1:1.5
                if tp_distance / sl_distance < 1.5:
                    logger.debug(f"Poor risk/reward ratio: {tp_distance/sl_distance:.2f}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking signal quality: {e}")
            return True  # Allow by default on error
    
    def is_kill_switch_active(self) -> bool:
        """Check if kill switch is active"""
        self.config = self._get_risk_config()
        return self.config.kill_switch_enabled
    
    def set_kill_switch(self, enabled: bool):
        """Set kill switch state"""
        self.config.kill_switch_enabled = enabled
        self.db.commit()
        logger.info(f"Kill switch {'enabled' if enabled else 'disabled'}")
    
    def get_daily_loss_limit(self) -> float:
        """Get current daily loss limit"""
        return self.config.daily_loss_limit
    
    def set_daily_loss_limit(self, limit: float):
        """Set daily loss limit"""
        self.config.daily_loss_limit = limit
        self.db.commit()
        logger.info(f"Daily loss limit set to ${limit}")
    
    def get_current_daily_loss(self) -> float:
        """Get current daily loss (estimated)"""
        try:
            today = datetime.utcnow().date()
            today_signals = self.db.query(Signal).filter(
                Signal.issued_at >= datetime.combine(today, datetime.min.time()),
                Signal.blocked_by_risk == False
            ).count()
            
            # Simplified estimation
            return today_signals * 50  # Assume $50 avg loss per signal
            
        except Exception:
            return 0.0
    
    def is_volatility_guard_active(self) -> bool:
        """Check if volatility guard is active"""
        return self.config.volatility_guard_enabled
    
    def get_risk_statistics(self) -> Dict[str, Any]:
        """Get comprehensive risk statistics"""
        try:
            today = datetime.utcnow().date()
            week_ago = today - timedelta(days=7)
            
            # Today's statistics
            today_signals = self.db.query(Signal).filter(
                Signal.issued_at >= datetime.combine(today, datetime.min.time())
            ).all()
            
            # Weekly statistics
            week_signals = self.db.query(Signal).filter(
                Signal.issued_at >= datetime.combine(week_ago, datetime.min.time())
            ).all()
            
            stats = {
                'kill_switch_enabled': self.config.kill_switch_enabled,
                'daily_loss_limit': self.config.daily_loss_limit,
                'volatility_guard_enabled': self.config.volatility_guard_enabled,
                'volatility_threshold': self.config.volatility_threshold,
                'max_daily_signals': self.config.max_daily_signals,
                'today_signals_count': len(today_signals),
                'today_blocked_count': len([s for s in today_signals if s.blocked_by_risk]),
                'week_signals_count': len(week_signals),
                'week_blocked_count': len([s for s in week_signals if s.blocked_by_risk]),
                'estimated_daily_loss': self.get_current_daily_loss()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting risk statistics: {e}")
            return {}
