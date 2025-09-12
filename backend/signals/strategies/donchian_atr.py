"""
Donchian Channel + ATR Strategy
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import talib as ta

from ..utils import calculate_sl_tp, calculate_atr
from ...logs.logger import get_logger

logger = get_logger(__name__)

class DonchianATRStrategy:
    """Donchian Channel Breakout with ATR Trailing Stop"""
    
    def generate_signal(self, data: pd.DataFrame, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate signal based on Donchian Channel breakout with ATR filter
        
        Strategy Logic:
        - Buy when price breaks above Donchian upper channel
        - Sell when price breaks below Donchian lower channel
        - Optional SuperTrend filter
        """
        try:
            donchian_period = config.get('donchian_period', 20)
            atr_period = config.get('atr_period', 14)
            atr_multiplier = config.get('atr_multiplier', 2.0)
            use_supertrend = config.get('use_supertrend', True)
            min_confidence = config.get('min_confidence', 0.65)
            
            if len(data) < max(donchian_period, atr_period) + 5:
                return None
            
            # Calculate Donchian Channels
            high_prices = data['high'].values
            low_prices = data['low'].values
            close_prices = data['close'].values
            
            upper_channel = ta.MAX(high_prices, timeperiod=donchian_period)
            lower_channel = ta.MIN(low_prices, timeperiod=donchian_period)
            
            # Calculate ATR
            atr_values = calculate_atr(data, atr_period)
            
            # Get current values
            current_price = close_prices[-1]
            current_high = high_prices[-1]
            current_low = low_prices[-1]
            prev_high = high_prices[-2]
            prev_low = low_prices[-2]
            current_upper = upper_channel[-1]
            current_lower = lower_channel[-1]
            current_atr = atr_values[-1]
            
            # Check for NaN values
            if (np.isnan(current_upper) or np.isnan(current_lower) or np.isnan(current_atr)):
                return None
            
            action = None
            confidence = 0.0
            
            # Bullish breakout - compare against previous channel value
            if current_high > upper_channel[-2] and prev_high <= upper_channel[-2]:
                action = "BUY"
                
                # Calculate confidence based on breakout strength
                breakout_strength = (current_high - upper_channel[-2]) / current_atr
                volatility_factor = min(current_atr / current_price, 0.01) * 100  # Normalize
                confidence = min(0.6 + breakout_strength * 0.2 + volatility_factor * 0.2, 1.0)
            
            # Bearish breakout - compare against previous channel value
            elif current_low < lower_channel[-2] and prev_low >= lower_channel[-2]:
                action = "SELL"
                
                # Calculate confidence
                breakout_strength = (lower_channel[-2] - current_low) / current_atr
                volatility_factor = min(current_atr / current_price, 0.01) * 100
                confidence = min(0.6 + breakout_strength * 0.2 + volatility_factor * 0.2, 1.0)
            
            # Apply SuperTrend filter if enabled
            if action and use_supertrend:
                supertrend_signal = self._calculate_supertrend_filter(data, atr_period)
                if supertrend_signal != action:
                    confidence *= 0.7  # Reduce confidence if SuperTrend disagrees
            
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
                    'donchian_upper': round(current_upper, 5),
                    'donchian_lower': round(current_lower, 5),
                    'atr': round(current_atr, 5)
                }
            }
            
            logger.debug(f"Donchian-ATR signal generated: {signal}")
            return signal
            
        except Exception as e:
            logger.error(f"Error generating Donchian-ATR signal: {e}")
            return None
    
    def _calculate_supertrend_filter(self, data: pd.DataFrame, atr_period: int) -> Optional[str]:
        """Calculate SuperTrend indicator for additional filtering"""
        try:
            high_prices = data['high'].values
            low_prices = data['low'].values
            close_prices = data['close'].values
            
            # Calculate SuperTrend
            atr_values = calculate_atr(data, atr_period)
            hl_avg = (high_prices + low_prices) / 2
            
            multiplier = 3.0
            upper_band = hl_avg + (multiplier * atr_values)
            lower_band = hl_avg - (multiplier * atr_values)
            
            # SuperTrend calculation
            supertrend = np.zeros_like(close_prices)
            direction = np.ones_like(close_prices)
            
            for i in range(1, len(close_prices)):
                if close_prices[i] <= lower_band[i]:
                    direction[i] = 1
                    supertrend[i] = lower_band[i]
                elif close_prices[i] >= upper_band[i]:
                    direction[i] = -1
                    supertrend[i] = upper_band[i]
                else:
                    direction[i] = direction[i-1]
                    if direction[i] == 1:
                        supertrend[i] = max(lower_band[i], supertrend[i-1])
                    else:
                        supertrend[i] = min(upper_band[i], supertrend[i-1])
            
            # Determine trend
            if direction[-1] == 1:
                return "BUY"
            else:
                return "SELL"
                
        except Exception:
            return None
