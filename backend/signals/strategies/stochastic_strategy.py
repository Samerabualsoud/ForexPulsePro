"""
Stochastic Oscillator Strategy - Momentum and Overbought/Oversold
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import talib as ta

from ..utils import calculate_sl_tp, enhance_signal_with_mt5_order_type
from ...logs.logger import get_logger

logger = get_logger(__name__)

class StochasticStrategy:
    """Stochastic Oscillator with %K and %D crossovers"""
    
    def generate_signal(self, data: pd.DataFrame, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate signal based on Stochastic oscillator crossovers
        
        Strategy Logic:
        - Buy when %K crosses above %D in oversold territory
        - Sell when %K crosses below %D in overbought territory
        - Confirm with momentum and volume analysis
        """
        try:
            k_period = config.get('stoch_k', 14)
            k_slow_period = config.get('stoch_k_slow', 3)
            d_period = config.get('stoch_d', 3)
            oversold_level = config.get('oversold', 20)
            overbought_level = config.get('overbought', 80)
            min_confidence = config.get('min_confidence', 0.8)  # User requested 80% minimum
            
            if len(data) < k_period + k_slow_period + d_period + 10:
                return None
                
            high_prices = data['high'].values
            low_prices = data['low'].values
            close_prices = data['close'].values
            
            # Calculate Stochastic oscillator
            slowk, slowd = ta.STOCH(
                high_prices, low_prices, close_prices,
                fastk_period=k_period,
                slowk_period=k_slow_period,
                slowk_matype=0,
                slowd_period=d_period,
                slowd_matype=0
            )
            
            # Get current and previous values
            current_k = slowk[-1]
            current_d = slowd[-1]
            prev_k = slowk[-2]
            prev_d = slowd[-2]
            current_price = close_prices[-1]
            
            # Check for NaN values
            if (np.isnan(current_k) or np.isnan(current_d)):
                return None
                
            action = None
            confidence = 0.0
            
            # Bullish crossover in oversold territory
            if (prev_k <= prev_d and current_k > current_d and current_k < oversold_level + 10):
                action = "BUY"
                
                # Calculate confidence based on position in oversold area and momentum
                oversold_factor = max(0, (oversold_level - current_k) / oversold_level)
                momentum = (current_k - prev_k) / 100
                crossover_strength = (current_k - current_d) / 100
                
                confidence = min(0.55 + oversold_factor * 0.25 + 
                               momentum * 0.1 + abs(crossover_strength) * 0.1, 1.0)
                
            # Bearish crossover in overbought territory  
            elif (prev_k >= prev_d and current_k < current_d and current_k > overbought_level - 10):
                action = "SELL"
                
                # Calculate confidence
                overbought_factor = max(0, (current_k - overbought_level) / (100 - overbought_level))
                momentum = (prev_k - current_k) / 100
                crossover_strength = (current_d - current_k) / 100
                
                confidence = min(0.55 + overbought_factor * 0.25 + 
                               momentum * 0.1 + abs(crossover_strength) * 0.1, 1.0)
            
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
            
            # Create base signal
            base_signal = {
                'action': action,
                'price': round(current_price, 5),
                'sl': round(sl, 5) if sl else None,
                'tp': round(tp, 5) if tp else None,
                'confidence': round(confidence, 2),
                'indicators': {
                    'stoch_k': round(current_k, 2),
                    'stoch_d': round(current_d, 2),
                    'oversold_level': oversold_level,
                    'overbought_level': overbought_level
                }
            }
            
            # Enhance with MT5 order type determination
            enhanced_signal = enhance_signal_with_mt5_order_type(
                signal_data=base_signal,
                data=data,
                config=config,
                strategy_type='momentum'  # Stochastic is a momentum/oscillator strategy
            )
            
            logger.debug(f"Stochastic signal generated: {enhanced_signal}")
            return enhanced_signal
            
        except Exception as e:
            logger.error(f"Error generating Stochastic signal: {e}")
            return None