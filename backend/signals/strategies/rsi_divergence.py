"""
RSI Divergence Strategy - Advanced momentum analysis with divergence detection
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import talib as ta

from ..utils import calculate_sl_tp, enhance_signal_with_mt5_order_type
from ...logs.logger import get_logger

logger = get_logger(__name__)

class RSIDivergenceStrategy:
    """RSI with Price Divergence Detection and Multi-timeframe Analysis"""
    
    def generate_signal(self, data: pd.DataFrame, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate signal based on RSI divergence and momentum analysis
        
        Strategy Logic:
        - Detect bullish divergence: Price makes lower lows, RSI makes higher lows
        - Detect bearish divergence: Price makes higher highs, RSI makes lower highs
        - Confirm with volume and momentum indicators
        """
        try:
            rsi_period = config.get('rsi_period', 14)
            lookback_period = config.get('lookback', 20)
            oversold_level = config.get('oversold', 30)
            overbought_level = config.get('overbought', 70)
            min_confidence = config.get('min_confidence', 0.8)  # User requested 80% minimum
            volume_filter = config.get('use_volume', True)
            
            if len(data) < rsi_period + lookback_period + 10:
                return None
                
            high_prices = data['high'].values
            low_prices = data['low'].values
            close_prices = data['close'].values
            volume = data.get('volume', pd.Series(np.ones(len(data)))).values
            
            # Calculate RSI and momentum indicators
            rsi = ta.RSI(close_prices, timeperiod=rsi_period)
            momentum = ta.MOM(close_prices, timeperiod=10)
            
            current_price = close_prices[-1]
            current_rsi = rsi[-1]
            
            if np.isnan(current_rsi):
                return None
                
            # Look for divergence patterns
            action = None
            confidence = 0.0
            divergence_strength = 0.0
            
            # Analyze recent highs and lows
            recent_data = lookback_period
            price_highs = []
            price_lows = []
            rsi_highs = []
            rsi_lows = []
            
            # Find local peaks and troughs
            for i in range(recent_data, len(close_prices)):
                if i >= 2 and i < len(close_prices) - 2:
                    # Local high
                    if (high_prices[i] > high_prices[i-1] and high_prices[i] > high_prices[i+1] and
                        high_prices[i] > high_prices[i-2] and high_prices[i] > high_prices[i+2]):
                        price_highs.append((i, high_prices[i]))
                        rsi_highs.append((i, rsi[i]))
                    
                    # Local low
                    if (low_prices[i] < low_prices[i-1] and low_prices[i] < low_prices[i+1] and
                        low_prices[i] < low_prices[i-2] and low_prices[i] < low_prices[i+2]):
                        price_lows.append((i, low_prices[i]))
                        rsi_lows.append((i, rsi[i]))
            
            # Check for bullish divergence
            if len(price_lows) >= 2 and len(rsi_lows) >= 2:
                last_price_low = price_lows[-1][1]
                prev_price_low = price_lows[-2][1] if len(price_lows) > 1 else price_lows[-1][1]
                last_rsi_low = rsi_lows[-1][1]
                prev_rsi_low = rsi_lows[-2][1] if len(rsi_lows) > 1 else rsi_lows[-1][1]
                
                # Bullish divergence: price lower low, RSI higher low
                if (last_price_low < prev_price_low and last_rsi_low > prev_rsi_low and 
                    current_rsi < oversold_level + 10):
                    action = "BUY"
                    divergence_strength = (last_rsi_low - prev_rsi_low) / (prev_price_low - last_price_low + 0.0001)
                    
                    # Calculate confidence
                    rsi_position = max(0, (oversold_level - current_rsi) / oversold_level)
                    momentum_strength = momentum[-1] / (abs(momentum[-1]) + 0.0001) if not np.isnan(momentum[-1]) else 0
                    
                    confidence = min(0.65 + abs(divergence_strength) * 0.15 + 
                                   rsi_position * 0.1 + momentum_strength * 0.1, 1.0)
            
            # Check for bearish divergence  
            if len(price_highs) >= 2 and len(rsi_highs) >= 2:
                last_price_high = price_highs[-1][1]
                prev_price_high = price_highs[-2][1] if len(price_highs) > 1 else price_highs[-1][1]
                last_rsi_high = rsi_highs[-1][1]
                prev_rsi_high = rsi_highs[-2][1] if len(rsi_highs) > 1 else rsi_highs[-1][1]
                
                # Bearish divergence: price higher high, RSI lower high
                if (last_price_high > prev_price_high and last_rsi_high < prev_rsi_high and 
                    current_rsi > overbought_level - 10):
                    action = "SELL"
                    divergence_strength = (prev_rsi_high - last_rsi_high) / (last_price_high - prev_price_high + 0.0001)
                    
                    # Calculate confidence
                    rsi_position = max(0, (current_rsi - overbought_level) / (100 - overbought_level))
                    momentum_strength = -momentum[-1] / (abs(momentum[-1]) + 0.0001) if not np.isnan(momentum[-1]) else 0
                    
                    confidence = min(0.65 + abs(divergence_strength) * 0.15 + 
                                   rsi_position * 0.1 + momentum_strength * 0.1, 1.0)
            
            # Apply volume filter
            if action and volume_filter:
                recent_volume = np.mean(volume[-5:])
                avg_volume = np.mean(volume[-20:])
                if recent_volume < avg_volume * 0.8:
                    confidence *= 0.8
                    
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
                    'rsi': round(current_rsi, 2),
                    'divergence_strength': round(divergence_strength, 4),
                    'momentum': round(momentum[-1], 6) if not np.isnan(momentum[-1]) else None,
                    'oversold': oversold_level,
                    'overbought': overbought_level
                }
            }
            
            # Enhance with MT5 order type determination
            enhanced_signal = enhance_signal_with_mt5_order_type(
                signal_data=base_signal,
                data=data,
                config=config,
                strategy_type='momentum'  # RSI divergence is a momentum-based strategy
            )
            
            logger.debug(f"RSI Divergence signal generated: {enhanced_signal}")
            return enhanced_signal
            
        except Exception as e:
            logger.error(f"Error generating RSI Divergence signal: {e}")
            return None