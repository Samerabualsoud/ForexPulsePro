"""
Market Regime Detection Module
Classifies market conditions as trending, ranging, or high-volatility
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

try:
    import talib as ta
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    ta = None

from ..logs.logger import get_logger
from ..models import MarketRegime
from ..database import engine

logger = get_logger(__name__)

class RegimeDetector:
    """Market regime detection using ADX and ATR indicators"""
    
    def __init__(self):
        self.adx_period = 14
        self.atr_period = 14
        self.trend_threshold = 25    # ADX > 25 = trending
        self.strong_trend_threshold = 40  # ADX > 40 = strong trend
        self.volatility_threshold = 0.015  # ATR/Price > 1.5% = high volatility
        
    def detect_regime(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Detect market regime for given symbol data
        
        Returns:
            Dict with regime classification and confidence
        """
        try:
            if not TALIB_AVAILABLE:
                return {
                    'regime': 'UNKNOWN',
                    'confidence': 0.0,
                    'adx': None,
                    'atr_ratio': None,
                    'reason': 'TA-Lib not available'
                }
            
            if len(data) < max(self.adx_period, self.atr_period) + 5:
                return {
                    'regime': 'UNKNOWN',
                    'confidence': 0.0,
                    'adx': None,
                    'atr_ratio': None,
                    'reason': 'Insufficient data'
                }
            
            # Calculate indicators
            high_prices = data['high'].values
            low_prices = data['low'].values 
            close_prices = data['close'].values
            
            # ADX for trend strength
            adx_values = ta.ADX(high_prices, low_prices, close_prices, timeperiod=self.adx_period)
            
            # ATR for volatility
            atr_values = ta.ATR(high_prices, low_prices, close_prices, timeperiod=self.atr_period)
            
            # Get current values
            current_adx = adx_values[-1]
            current_atr = atr_values[-1]
            current_price = close_prices[-1]
            
            # Check for NaN values
            if np.isnan(current_adx) or np.isnan(current_atr):
                return {
                    'regime': 'UNKNOWN',
                    'confidence': 0.0,
                    'adx': None,
                    'atr_ratio': None,
                    'reason': 'Invalid indicator values'
                }
            
            # Calculate volatility ratio
            atr_ratio = current_atr / current_price
            
            # Classify regime
            regime, confidence = self._classify_regime(current_adx, atr_ratio)
            
            return {
                'regime': regime,
                'confidence': float(confidence),
                'adx': float(round(current_adx, 2)),
                'atr_ratio': float(round(atr_ratio, 4)),
                'reason': f'ADX: {current_adx:.1f}, ATR Ratio: {atr_ratio:.3f}'
            }
            
        except Exception as e:
            logger.error(f"Error detecting regime for {symbol}: {e}")
            return {
                'regime': 'UNKNOWN',
                'confidence': 0.0,
                'adx': None,
                'atr_ratio': None,
                'reason': f'Error: {str(e)}'
            }
    
    def _classify_regime(self, adx: float, atr_ratio: float) -> Tuple[str, float]:
        """
        Classify market regime based on ADX and ATR ratio
        
        Returns:
            Tuple of (regime, confidence)
        """
        # High volatility check first
        if atr_ratio > self.volatility_threshold:
            confidence = min((atr_ratio - self.volatility_threshold) / self.volatility_threshold, 1.0)
            return 'HIGH_VOLATILITY', 0.6 + confidence * 0.4
        
        # Trend strength classification
        if adx >= self.strong_trend_threshold:
            # Strong trending market
            confidence = min((adx - self.strong_trend_threshold) / 20, 1.0)
            return 'STRONG_TRENDING', 0.8 + confidence * 0.2
            
        elif adx >= self.trend_threshold:
            # Moderate trending market
            confidence = (adx - self.trend_threshold) / (self.strong_trend_threshold - self.trend_threshold)
            return 'TRENDING', 0.6 + confidence * 0.2
            
        else:
            # Ranging market
            confidence = (self.trend_threshold - adx) / self.trend_threshold
            return 'RANGING', 0.5 + confidence * 0.3
    
    def store_regime(self, symbol: str, regime_data: Dict[str, Any], db: Session):
        """Store regime classification in database"""
        try:
            # Convert numpy types to native Python types for database storage
            confidence = regime_data['confidence']
            adx = regime_data.get('adx')
            atr_ratio = regime_data.get('atr_ratio')
            
            # Ensure values are native Python types, not numpy types
            if hasattr(confidence, 'item'):
                confidence = float(confidence.item())
            elif isinstance(confidence, np.floating):
                confidence = float(confidence)
            
            if adx is not None:
                if hasattr(adx, 'item'):
                    adx = float(adx.item())
                elif isinstance(adx, np.floating):
                    adx = float(adx)
                    
            if atr_ratio is not None:
                if hasattr(atr_ratio, 'item'):
                    atr_ratio = float(atr_ratio.item())
                elif isinstance(atr_ratio, np.floating):
                    atr_ratio = float(atr_ratio)
            
            regime = MarketRegime(
                symbol=symbol,
                regime=regime_data['regime'],
                confidence=confidence,
                adx=adx,
                atr_ratio=atr_ratio,
                detected_at=datetime.utcnow()
            )
            
            db.add(regime)
            db.commit()
            
            logger.debug(f"Stored regime for {symbol}: {regime_data['regime']} ({regime_data['confidence']:.2f})")
            
        except Exception as e:
            logger.error(f"Error storing regime for {symbol}: {e}")
            db.rollback()
    
    def get_latest_regime(self, symbol: str, db: Session) -> Optional[Dict[str, Any]]:
        """Get latest regime for symbol from database"""
        try:
            latest_regime = db.query(MarketRegime).filter(
                MarketRegime.symbol == symbol
            ).order_by(MarketRegime.detected_at.desc()).first()
            
            if latest_regime:
                return {
                    'regime': latest_regime.regime,
                    'confidence': latest_regime.confidence,
                    'adx': latest_regime.adx,
                    'atr_ratio': latest_regime.atr_ratio,
                    'detected_at': latest_regime.detected_at
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting latest regime for {symbol}: {e}")
            return None
    
    def is_regime_suitable_for_strategy(self, regime: str, strategy_name: str) -> bool:
        """
        Check if current regime is suitable for strategy
        
        Strategy suitability:
        - Trending strategies (EMA, MACD, Donchian) work best in TRENDING/STRONG_TRENDING
        - Mean reversion strategies (BB, RSI Divergence) work best in RANGING
        - Momentum strategies (Stochastic, Fibonacci) can work in most regimes but avoid HIGH_VOLATILITY
        """
        if regime == 'UNKNOWN':
            return True  # Allow when regime is unknown
            
        strategy_regime_map = {
            'ema_rsi': ['TRENDING', 'STRONG_TRENDING'],
            'macd_crossover': ['TRENDING', 'STRONG_TRENDING'],  
            'donchian_atr': ['TRENDING', 'STRONG_TRENDING'],
            'meanrev_bb': ['RANGING'],
            'rsi_divergence': ['RANGING', 'TRENDING'],
            'stochastic': ['TRENDING', 'RANGING'],
            'fibonacci': ['TRENDING', 'RANGING']
        }
        
        suitable_regimes = strategy_regime_map.get(strategy_name, ['TRENDING', 'RANGING'])
        return regime in suitable_regimes

# Global regime detector instance
regime_detector = RegimeDetector()