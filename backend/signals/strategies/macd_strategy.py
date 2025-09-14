"""
MACD Strategy - Advanced momentum-based trading
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import talib as ta

from ..utils import calculate_sl_tp
from ...logs.logger import get_logger

logger = get_logger(__name__)

class MACDStrategy:
    """MACD Signal Line Crossover with Histogram Divergence"""
    
    def generate_signal(self, data: pd.DataFrame, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate signal based on MACD crossover and histogram analysis
        
        Strategy Logic:
        - Buy when MACD line crosses above signal line with positive momentum
        - Sell when MACD line crosses below signal line with negative momentum
        - Confirm with histogram strength and volume
        """
        try:
            fast_period = config.get('macd_fast', 12)
            slow_period = config.get('macd_slow', 26)
            signal_period = config.get('macd_signal', 9)
            min_confidence = config.get('min_confidence', 0.8)  # User requested 80% minimum
            use_histogram_filter = config.get('use_histogram', True)
            
            if len(data) < max(slow_period, signal_period) + 10:
                return None
                
            close_prices = data['close'].values
            
            # Calculate MACD components
            macd, macd_signal, macd_histogram = ta.MACD(
                close_prices, 
                fastperiod=fast_period,
                slowperiod=slow_period, 
                signalperiod=signal_period
            )
            
            # Get current and previous values
            current_macd = macd[-1]
            current_signal = macd_signal[-1]
            current_histogram = macd_histogram[-1]
            prev_macd = macd[-2]
            prev_signal = macd_signal[-2]
            prev_histogram = macd_histogram[-2]
            current_price = close_prices[-1]
            
            # Check for NaN values
            if (np.isnan(current_macd) or np.isnan(current_signal) or np.isnan(current_histogram)):
                return None
                
            action = None
            confidence = 0.0
            
            # Bullish MACD crossover
            if (prev_macd <= prev_signal and current_macd > current_signal):
                action = "BUY"
                
                # Calculate confidence based on histogram strength and momentum
                histogram_strength = abs(current_histogram) / (abs(current_macd) + 0.0001)
                momentum_factor = (current_macd - prev_macd) / abs(prev_macd + 0.0001)
                signal_strength = (current_macd - current_signal) / abs(current_signal + 0.0001)
                
                confidence = min(0.6 + histogram_strength * 0.2 + 
                               abs(momentum_factor) * 0.1 + 
                               abs(signal_strength) * 0.1, 1.0)
                
            # Bearish MACD crossover
            elif (prev_macd >= prev_signal and current_macd < current_signal):
                action = "SELL"
                
                # Calculate confidence
                histogram_strength = abs(current_histogram) / (abs(current_macd) + 0.0001)
                momentum_factor = (prev_macd - current_macd) / abs(prev_macd + 0.0001)
                signal_strength = (current_signal - current_macd) / abs(current_signal + 0.0001)
                
                confidence = min(0.6 + histogram_strength * 0.2 + 
                               abs(momentum_factor) * 0.1 + 
                               abs(signal_strength) * 0.1, 1.0)
            
            # Apply histogram filter if enabled
            if action and use_histogram_filter:
                # Require histogram to be strengthening in signal direction
                if action == "BUY" and current_histogram < prev_histogram:
                    confidence *= 0.7
                elif action == "SELL" and current_histogram > prev_histogram:
                    confidence *= 0.7
                    
            # Check minimum confidence
            if action is None or confidence < min_confidence:
                return None
                
            # Calculate stop loss and take profit
            sl, tp = calculate_sl_tp(
                price=current_price,
                action=action,
                data=data,
                config=config
            )
            
            signal = {
                'action': action,
                'price': round(current_price, 5),
                'sl': round(sl, 5) if sl else None,
                'tp': round(tp, 5) if tp else None,
                'confidence': round(confidence, 2),
                'indicators': {
                    'macd': round(current_macd, 6),
                    'macd_signal': round(current_signal, 6),
                    'histogram': round(current_histogram, 6),
                    'momentum': round(momentum_factor, 4) if 'momentum_factor' in locals() else None
                }
            }
            
            logger.debug(f"MACD signal generated: {signal}")
            return signal
            
        except Exception as e:
            logger.error(f"Error generating MACD signal: {e}")
            return None