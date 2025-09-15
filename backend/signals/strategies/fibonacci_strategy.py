"""
Fibonacci Retracement Strategy - Advanced support/resistance levels
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import talib as ta

from ..utils import calculate_sl_tp, enhance_signal_with_mt5_order_type
from ...logs.logger import get_logger

logger = get_logger(__name__)

class FibonacciStrategy:
    """Fibonacci Retracement with Golden Ratio Analysis"""
    
    def generate_signal(self, data: pd.DataFrame, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate signal based on Fibonacci retracement levels and price action
        
        Strategy Logic:
        - Identify recent swing high/low
        - Calculate Fibonacci retracement levels (23.6%, 38.2%, 50%, 61.8%, 78.6%)
        - Buy at support levels in uptrend, sell at resistance in downtrend
        - Confirm with momentum and volume
        """
        try:
            swing_period = config.get('swing_period', 20)
            fib_tolerance = config.get('fib_tolerance', 0.0005)  # Price tolerance near fib levels
            min_confidence = config.get('min_confidence', 0.8)  # User requested 80% minimum
            trend_period = config.get('trend_period', 50)
            
            if len(data) < max(swing_period, trend_period) + 10:
                return None
                
            high_prices = data['high'].values
            low_prices = data['low'].values
            close_prices = data['close'].values
            
            current_price = close_prices[-1]
            
            # Determine overall trend using EMA
            ema_fast = ta.EMA(close_prices, timeperiod=20)
            ema_slow = ta.EMA(close_prices, timeperiod=50)
            
            if np.isnan(ema_fast[-1]) or np.isnan(ema_slow[-1]):
                return None
                
            trend = "UP" if ema_fast[-1] > ema_slow[-1] else "DOWN"
            
            # Find recent swing high and low
            recent_high = np.max(high_prices[-swing_period:])
            recent_low = np.min(low_prices[-swing_period:])
            swing_range = recent_high - recent_low
            
            if swing_range == 0:
                return None
                
            # Calculate Fibonacci levels
            fib_levels = {
                '0.0': recent_low,
                '23.6': recent_low + swing_range * 0.236,
                '38.2': recent_low + swing_range * 0.382,
                '50.0': recent_low + swing_range * 0.500,
                '61.8': recent_low + swing_range * 0.618,
                '78.6': recent_low + swing_range * 0.786,
                '100.0': recent_high
            }
            
            # Find closest Fibonacci level
            closest_level = None
            closest_distance = float('inf')
            closest_ratio = None
            
            for ratio, level in fib_levels.items():
                distance = abs(current_price - level)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_level = level
                    closest_ratio = ratio
                    
            # Check if price is near a Fibonacci level
            if closest_distance > current_price * fib_tolerance:
                return None
                
            action = None
            confidence = 0.0
            
            # RSI for momentum confirmation
            rsi = ta.RSI(close_prices, timeperiod=14)
            current_rsi = rsi[-1]
            
            if np.isnan(current_rsi):
                return None
                
            # Buy signals at support levels in uptrend
            if trend == "UP" and closest_ratio in ['23.6', '38.2', '50.0']:
                if current_rsi < 55:  # Not overbought
                    action = "BUY"
                    
                    # Calculate confidence based on Fibonacci level importance and momentum
                    fib_strength = {'23.6': 0.7, '38.2': 0.8, '50.0': 0.9}[closest_ratio]
                    trend_strength = min((ema_fast[-1] - ema_slow[-1]) / ema_slow[-1] * 100, 0.1)
                    rsi_factor = (55 - current_rsi) / 55
                    
                    confidence = min(0.6 + fib_strength * 0.2 + 
                                   trend_strength + rsi_factor * 0.1, 1.0)
                    
            # Sell signals at resistance levels in downtrend  
            elif trend == "DOWN" and closest_ratio in ['61.8', '78.6', '100.0']:
                if current_rsi > 45:  # Not oversold
                    action = "SELL"
                    
                    # Calculate confidence
                    fib_strength = {'61.8': 0.8, '78.6': 0.9, '100.0': 0.7}[closest_ratio]
                    trend_strength = min((ema_slow[-1] - ema_fast[-1]) / ema_slow[-1] * 100, 0.1)
                    rsi_factor = (current_rsi - 45) / 55
                    
                    confidence = min(0.6 + fib_strength * 0.2 + 
                                   trend_strength + rsi_factor * 0.1, 1.0)
            
            # Check minimum confidence
            if action is None or confidence < min_confidence:
                return None
                
            # Calculate stop loss and take profit with Fibonacci levels
            if action == "BUY":
                # Set SL below next lower Fibonacci level
                sl_levels = [v for k, v in fib_levels.items() if v < current_price]
                sl_price = max(sl_levels) if sl_levels else current_price * 0.995
                
                # Set TP at next higher Fibonacci level
                tp_levels = [v for k, v in fib_levels.items() if v > current_price]
                tp_price = min(tp_levels) if tp_levels else current_price * 1.015
                
            else:  # SELL
                # Set SL above next higher Fibonacci level
                sl_levels = [v for k, v in fib_levels.items() if v > current_price]
                sl_price = min(sl_levels) if sl_levels else current_price * 1.005
                
                # Set TP at next lower Fibonacci level
                tp_levels = [v for k, v in fib_levels.items() if v < current_price]
                tp_price = max(tp_levels) if tp_levels else current_price * 0.985
            
            # Create base signal
            base_signal = {
                'action': action,
                'price': round(current_price, 5),
                'sl': round(sl_price, 5),
                'tp': round(tp_price, 5),
                'confidence': round(confidence, 2),
                'indicators': {
                    'fib_level': closest_ratio,
                    'fib_price': round(closest_level, 5),
                    'trend': trend,
                    'rsi': round(current_rsi, 2),
                    'swing_high': round(recent_high, 5),
                    'swing_low': round(recent_low, 5)
                }
            }
            
            # Enhance with MT5 order type determination
            # Note: Use current_price instead of closest_level for price comparison
            base_signal['price'] = round(current_price, 5)
            enhanced_signal = enhance_signal_with_mt5_order_type(
                signal_data=base_signal,
                data=data,
                config=config,
                strategy_type='retracement'  # Fibonacci is a retracement/support-resistance strategy
            )
            
            # Restore the Fibonacci-calculated SL/TP levels
            enhanced_signal['sl'] = round(sl_price, 5)
            enhanced_signal['tp'] = round(tp_price, 5)
            
            logger.debug(f"Fibonacci signal generated: {enhanced_signal}")
            return enhanced_signal
            
        except Exception as e:
            logger.error(f"Error generating Fibonacci signal: {e}")
            return None