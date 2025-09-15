"""
EMA + RSI Strategy
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import talib as ta

from ..utils import calculate_sl_tp, enhance_signal_with_mt5_order_type
from ...logs.logger import get_logger

logger = get_logger(__name__)

class EMAStragey:
    """EMA Crossover with RSI Filter Strategy"""
    
    def generate_signal(self, data: pd.DataFrame, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate signal based on EMA crossover with RSI filter
        
        Strategy Logic:
        - Buy when EMA fast crosses above EMA slow and RSI > 50
        - Sell when EMA fast crosses below EMA slow and RSI < 50
        """
        try:
            if len(data) < max(config.get('ema_slow', 26), config.get('rsi_period', 14)) + 5:
                return None
            
            # Extract configuration
            ema_fast = config.get('ema_fast', 12)
            ema_slow = config.get('ema_slow', 26)
            rsi_period = config.get('rsi_period', 14)
            rsi_buy_threshold = config.get('rsi_buy_threshold', 45)  # More lenient
            rsi_sell_threshold = config.get('rsi_sell_threshold', 55)  # More lenient
            min_confidence = config.get('min_confidence', 0.8)  # User requested 80% minimum
            
            # Calculate indicators
            close_prices = np.asarray(data['close'].values, dtype=np.float64)
            ema_fast_values = ta.EMA(close_prices, timeperiod=ema_fast)
            ema_slow_values = ta.EMA(close_prices, timeperiod=ema_slow)
            rsi_values = ta.RSI(close_prices, timeperiod=rsi_period)
            
            # Get latest values
            current_ema_fast = ema_fast_values[-1]
            current_ema_slow = ema_slow_values[-1]
            prev_ema_fast = ema_fast_values[-2]
            prev_ema_slow = ema_slow_values[-2]
            current_rsi = rsi_values[-1]
            current_price = close_prices[-1]
            
            # Check for NaN values
            if (np.isnan(current_ema_fast) or np.isnan(current_ema_slow) or 
                np.isnan(prev_ema_fast) or np.isnan(prev_ema_slow) or np.isnan(current_rsi)):
                return None
            
            # Detect crossover
            action = None
            confidence = 0.0
            
            # Bullish crossover - more flexible conditions
            if (current_ema_fast > current_ema_slow and 
                current_rsi > rsi_buy_threshold and
                current_ema_fast > prev_ema_fast):  # Upward momentum
                action = "BUY"
                
                # Simplified confidence calculation
                rsi_strength = max(0, (current_rsi - 45) / 55)  # 0 to 1
                ema_momentum = max(0, (current_ema_fast - prev_ema_fast) / current_price * 1000)
                confidence = min(0.4 + rsi_strength * 0.3 + ema_momentum * 0.3, 0.95)
            
            # Bearish crossover - more flexible conditions
            elif (current_ema_fast < current_ema_slow and 
                  current_rsi < rsi_sell_threshold and
                  current_ema_fast < prev_ema_fast):  # Downward momentum
                action = "SELL"
                
                # Calculate confidence
                rsi_strength = min((50 - current_rsi) / 50, 1.0)  # 0 to 1
                ema_separation = (current_ema_slow - current_ema_fast) / current_price
                confidence = min(0.5 + rsi_strength * 0.3 + abs(ema_separation) * 10000, 1.0)
            
            # No signal if action is flat or confidence too low
            if action is None or confidence < min_confidence:
                return None
            
            # Calculate stop loss and take profit
            sl, tp = calculate_sl_tp(
                price=current_price,
                action=action,
                data=data,
                config=config
            )
            
            # Create base signal
            base_signal = {
                'action': action,
                'price': round(current_price, 5),
                'sl': round(sl, 5) if sl else None,
                'tp': round(tp, 5) if tp else None,
                'confidence': round(confidence, 2),
                'indicators': {
                    'ema_fast': round(current_ema_fast, 5),
                    'ema_slow': round(current_ema_slow, 5),
                    'rsi': round(current_rsi, 2)
                }
            }
            
            # Enhance with MT5 order type determination
            enhanced_signal = enhance_signal_with_mt5_order_type(
                signal_data=base_signal,
                data=data,
                config=config,
                strategy_type='momentum'  # EMA crossover is a momentum strategy
            )
            
            logger.debug(f"EMA-RSI signal generated: {enhanced_signal}")
            return enhanced_signal
            
        except Exception as e:
            logger.error(f"Error generating EMA-RSI signal: {e}")
            return None
