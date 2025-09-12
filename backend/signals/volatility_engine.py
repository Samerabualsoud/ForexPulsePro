"""
Advanced Volatility Analysis and Dynamic Position Sizing Engine
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import structlog
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning("TA-Lib not available, using pure Python fallback")

logger = structlog.get_logger(__name__)

class VolatilityAnalyzer:
    """Professional volatility analysis for risk-adjusted position sizing"""
    
    def __init__(self):
        self.atr_period = 14
        self.volatility_lookback = 30
        self.volatility_cache = {}
        
    def calculate_atr(self, ohlc_data: List[Dict]) -> float:
        """Calculate Average True Range for volatility measurement"""
        if len(ohlc_data) < self.atr_period:
            return 0.01  # Default volatility
        
        try:
            df = pd.DataFrame(ohlc_data)
            
            if TALIB_AVAILABLE:
                high = df['high'].values
                low = df['low'].values  
                close = df['close'].values
                atr = talib.ATR(high, low, close, timeperiod=self.atr_period)
                return float(atr[-1]) if not np.isnan(atr[-1]) else 0.01
            else:
                # Pure Python ATR fallback
                return self._calculate_atr_fallback(df)
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return 0.01
    
    def _calculate_atr_fallback(self, df: pd.DataFrame) -> float:
        """Pure Python ATR calculation fallback"""
        try:
            # Calculate True Range
            df['prev_close'] = df['close'].shift(1)
            df['tr1'] = df['high'] - df['low']
            df['tr2'] = abs(df['high'] - df['prev_close'])
            df['tr3'] = abs(df['low'] - df['prev_close'])
            df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            
            # Calculate ATR as simple moving average of True Range
            atr = df['true_range'].rolling(window=self.atr_period).mean()
            return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.01
            
        except Exception as e:
            logger.error(f"Error in ATR fallback calculation: {e}")
            return 0.01
    
    def calculate_volatility_regime(self, symbol: str, ohlc_data: List[Dict]) -> Dict[str, Any]:
        """
        Determine current volatility regime for dynamic risk management
        
        Returns:
            Dict with volatility classification and metrics
        """
        try:
            if len(ohlc_data) < self.volatility_lookback:
                return self._default_volatility_regime()
            
            df = pd.DataFrame(ohlc_data)
            closes = df['close'].values
            
            # Calculate returns
            returns = np.diff(np.log(closes))
            
            # Current volatility (rolling 14-period)
            current_vol = np.std(returns[-14:]) * np.sqrt(252) if len(returns) >= 14 else np.std(returns) * np.sqrt(252)
            
            # Long-term average volatility
            avg_vol = np.std(returns) * np.sqrt(252)
            
            # Volatility percentile (where current vol stands relative to history)
            vol_series = []
            for i in range(14, len(returns)):
                period_vol = np.std(returns[i-14:i]) * np.sqrt(252)
                vol_series.append(period_vol)
            
            vol_percentile = np.percentile(vol_series, [25, 50, 75]) if vol_series else [current_vol] * 3
            
            # Classify volatility regime
            if current_vol > vol_percentile[2]:
                regime = "HIGH"
                risk_multiplier = 0.6  # Reduce position size in high vol
            elif current_vol < vol_percentile[0]:
                regime = "LOW" 
                risk_multiplier = 1.2  # Increase position size in low vol
            else:
                regime = "NORMAL"
                risk_multiplier = 1.0
            
            # Calculate ATR for stop loss sizing
            atr = self.calculate_atr(ohlc_data)
            
            regime_data = {
                "symbol": symbol,
                "current_volatility": round(current_vol * 100, 2),  # Convert to percentage
                "avg_volatility": round(avg_vol * 100, 2),
                "volatility_regime": regime,
                "risk_multiplier": risk_multiplier,
                "atr": atr,
                "vol_percentile": {
                    "25th": round(vol_percentile[0] * 100, 2),
                    "50th": round(vol_percentile[1] * 100, 2),
                    "75th": round(vol_percentile[2] * 100, 2)
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache result
            self.volatility_cache[symbol] = regime_data
            
            logger.debug(f"Volatility regime for {symbol}: {regime} ({current_vol:.1%})")
            return regime_data
            
        except Exception as e:
            logger.error(f"Error calculating volatility regime: {e}")
            return self._default_volatility_regime()
    
    def _default_volatility_regime(self) -> Dict[str, Any]:
        """Default volatility regime when calculation fails"""
        return {
            "current_volatility": 1.5,
            "volatility_regime": "NORMAL",
            "risk_multiplier": 1.0,
            "atr": 0.01,
            "error": "Insufficient data"
        }

class DynamicPositionSizer:
    """Advanced position sizing based on volatility and risk parameters"""
    
    def __init__(self):
        self.base_risk_percent = 2.0  # 2% risk per trade
        self.max_risk_percent = 5.0   # Maximum risk allowed
        self.min_risk_percent = 0.5   # Minimum risk allowed
        
    def calculate_position_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss: float,
        volatility_data: Dict[str, Any],
        confidence: float,
        strategy_multiplier: float = 1.0
    ) -> Dict[str, Any]:
        """
        Calculate optimal position size based on multiple risk factors
        
        Args:
            account_balance: Trading account balance
            entry_price: Planned entry price
            stop_loss: Stop loss price
            volatility_data: Volatility analysis data
            confidence: Signal confidence (0.0-1.0)
            strategy_multiplier: Strategy performance multiplier
            
        Returns:
            Position sizing recommendations
        """
        try:
            # Base calculations
            risk_distance = abs(entry_price - stop_loss)
            if risk_distance == 0:
                logger.warning("Zero risk distance detected")
                return self._default_position_size(account_balance)
            
            # Dynamic risk adjustment factors
            volatility_multiplier = volatility_data.get('risk_multiplier', 1.0)
            confidence_multiplier = 0.5 + (confidence * 0.5)  # Scale from 0.5 to 1.0
            
            # Calculate adjusted risk percentage
            adjusted_risk = (
                self.base_risk_percent *
                volatility_multiplier *
                confidence_multiplier *
                strategy_multiplier
            )
            
            # Apply bounds
            adjusted_risk = max(self.min_risk_percent, min(self.max_risk_percent, adjusted_risk))
            
            # Calculate position size
            risk_amount = account_balance * (adjusted_risk / 100)
            position_size = risk_amount / risk_distance
            
            # Convert to lots (assuming 100k per lot for forex)
            lot_size = position_size / 100000
            
            # Round to standard lot sizes
            if lot_size >= 1.0:
                lots = round(lot_size, 1)
            elif lot_size >= 0.1:
                lots = round(lot_size, 2)
            else:
                lots = max(0.01, round(lot_size, 2))  # Minimum micro lot
            
            # Calculate actual risk with rounded lots
            actual_position_value = lots * 100000
            actual_risk_amount = actual_position_value * risk_distance
            actual_risk_percent = (actual_risk_amount / account_balance) * 100
            
            sizing_data = {
                "recommended_lots": lots,
                "position_value": round(actual_position_value, 2),
                "risk_amount": round(actual_risk_amount, 2),
                "risk_percent": round(actual_risk_percent, 2),
                "risk_distance_pips": round(risk_distance * 10000, 1),  # Convert to pips
                "adjustments": {
                    "base_risk": self.base_risk_percent,
                    "volatility_adj": volatility_multiplier,
                    "confidence_adj": confidence_multiplier,
                    "strategy_adj": strategy_multiplier,
                    "final_risk": round(adjusted_risk, 2)
                },
                "volatility_regime": volatility_data.get('volatility_regime', 'UNKNOWN'),
                "calculated_at": datetime.now().isoformat()
            }
            
            logger.debug(f"Position size calculated: {lots} lots, {actual_risk_percent:.1f}% risk")
            return sizing_data
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return self._default_position_size(account_balance)
    
    def _default_position_size(self, account_balance: float) -> Dict[str, Any]:
        """Default position size when calculation fails"""
        default_lots = 0.1
        return {
            "recommended_lots": default_lots,
            "position_value": 10000,
            "risk_amount": account_balance * 0.02,
            "risk_percent": 2.0,
            "error": "Calculation failed, using defaults"
        }
    
    def calculate_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate Kelly Criterion for optimal position sizing
        
        Returns:
            Optimal fraction of capital to risk (0.0 to 1.0)
        """
        if win_rate <= 0 or win_rate >= 1 or avg_loss <= 0:
            return 0.02  # Default 2%
        
        try:
            # Kelly formula: f = (bp - q) / b
            # where b = odds received on wager (avg_win / avg_loss)
            # p = probability of winning
            # q = probability of losing (1 - p)
            
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - win_rate
            
            kelly_fraction = (b * p - q) / b
            
            # Apply fractional Kelly (quarter Kelly) for safety
            safe_kelly = max(0, min(0.25, kelly_fraction * 0.25))
            
            logger.debug(f"Kelly criterion: {kelly_fraction:.3f}, Safe Kelly: {safe_kelly:.3f}")
            return safe_kelly
            
        except Exception as e:
            logger.error(f"Error calculating Kelly criterion: {e}")
            return 0.02

class RiskAdjustedSignalFilter:
    """Filter and adjust signals based on comprehensive risk analysis"""
    
    def __init__(self):
        self.volatility_analyzer = VolatilityAnalyzer()
        self.position_sizer = DynamicPositionSizer()
    
    def process_signal_with_risk_adjustment(
        self,
        signal: Dict[str, Any],
        market_data: List[Dict],
        account_balance: float = 10000,
        strategy_performance: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive signal processing with risk adjustments
        
        Returns:
            Enhanced signal with risk-adjusted parameters
        """
        try:
            symbol = signal.get('symbol', 'UNKNOWN')
            
            # Analyze current volatility regime
            volatility_data = self.volatility_analyzer.calculate_volatility_regime(symbol, market_data)
            
            # Get strategy performance multiplier
            strategy_multiplier = 1.0
            if strategy_performance:
                strategy_multiplier = strategy_performance.get('performance_multiplier', 1.0)
            
            # Calculate optimal position size
            position_data = self.position_sizer.calculate_position_size(
                account_balance=account_balance,
                entry_price=signal.get('price', 0),
                stop_loss=signal.get('sl', signal.get('price', 0)),
                volatility_data=volatility_data,
                confidence=signal.get('confidence', 0.5),
                strategy_multiplier=strategy_multiplier
            )
            
            # Risk-adjusted signal filtering
            should_trade = self._should_trade_signal(signal, volatility_data, position_data)
            
            # Enhance original signal
            enhanced_signal = signal.copy()
            enhanced_signal.update({
                'volatility_analysis': volatility_data,
                'position_sizing': position_data,
                'risk_adjusted': True,
                'trade_recommendation': 'TAKE' if should_trade else 'SKIP',
                'risk_score': self._calculate_risk_score(volatility_data, position_data),
                'processing_timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Risk-adjusted signal for {symbol}: {'TAKE' if should_trade else 'SKIP'}")
            return enhanced_signal
            
        except Exception as e:
            logger.error(f"Error processing signal with risk adjustment: {e}")
            signal['error'] = str(e)
            return signal
    
    def _should_trade_signal(
        self,
        signal: Dict[str, Any],
        volatility_data: Dict[str, Any],
        position_data: Dict[str, Any]
    ) -> bool:
        """Determine if signal should be traded based on risk analysis"""
        
        # Skip if volatility is too extreme
        volatility_regime = volatility_data.get('volatility_regime', 'NORMAL')
        if volatility_regime == 'HIGH':
            current_vol = volatility_data.get('current_volatility', 0)
            if current_vol > 5.0:  # More than 5% daily volatility
                logger.info(f"Skipping signal due to extreme volatility: {current_vol:.1f}%")
                return False
        
        # Skip if risk is too high
        risk_percent = position_data.get('risk_percent', 0)
        if risk_percent > 3.0:  # More than 3% risk
            logger.info(f"Skipping signal due to high risk: {risk_percent:.1f}%")
            return False
        
        # Skip if confidence is too low
        confidence = signal.get('confidence', 0)
        min_confidence = 0.6 if volatility_regime == 'HIGH' else 0.5
        if confidence < min_confidence:
            logger.info(f"Skipping signal due to low confidence: {confidence:.2f}")
            return False
        
        return True
    
    def _calculate_risk_score(self, volatility_data: Dict, position_data: Dict) -> float:
        """Calculate overall risk score (0-10, higher = riskier)"""
        try:
            base_score = 5.0  # Neutral risk
            
            # Adjust for volatility
            vol_regime = volatility_data.get('volatility_regime', 'NORMAL')
            if vol_regime == 'HIGH':
                base_score += 2.0
            elif vol_regime == 'LOW':
                base_score -= 1.0
            
            # Adjust for position risk
            risk_percent = position_data.get('risk_percent', 2.0)
            if risk_percent > 3.0:
                base_score += 1.5
            elif risk_percent < 1.0:
                base_score -= 1.0
            
            return max(0.0, min(10.0, base_score))
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 5.0  # Default moderate risk

# Global instances
volatility_analyzer = VolatilityAnalyzer()
position_sizer = DynamicPositionSizer()
risk_filter = RiskAdjustedSignalFilter()