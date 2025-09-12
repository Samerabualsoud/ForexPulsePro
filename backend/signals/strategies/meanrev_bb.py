"""
Mean Reversion + Bollinger Bands Strategy
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import talib as ta

from ..utils import calculate_sl_tp
from ...logs.logger import get_logger

logger = get_logger(__name__)

class MeanReversionBBStrategy:
    """Bollinger Bands Mean Reversion Strategy with ADX Filter"""
    
    def generate_signal(self, data: pd.DataFrame, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate signal based on Bollinger Bands mean reversion
        
        Strategy Logic:
        - Buy when price touches lower BB and shows reversal (oversold)
        - Sell when price touches upper BB and shows reversal (overbought)
        - ADX filter to avoid ranging markets
        """
        try:
            bb_period = config.get('bb_period', 20)
            bb_std = config.get('bb_std', 2.0)
            adx_period = config.get('adx_period', 14)
            adx_threshold = config.get('adx_threshold', 25)
            zscore_threshold = config.get('zscore_threshold', 2.0)
            min_confidence = config.get('min_confidence', 0.7)
            
            if len(data) < max(bb_period, adx_period) + 5:
                return None
            
            # Calculate Bollinger Bands
            close_prices = data['close'].values
            high_prices = data['high'].values
            low_prices = data['low'].values
            
            bb_upper, bb_middle, bb_lower = ta.BBANDS(
                close_prices, 
                timeperiod=bb_period, 
                nbdevup=bb_std, 
                nbdevdn=bb_std
            )
            
            # Calculate ADX for trend strength
            adx_values = ta.ADX(high_prices, low_prices, close_prices, timeperiod=adx_period)
            
            # Calculate Z-Score
            rolling_mean = ta.SMA(close_prices, timeperiod=bb_period)
            rolling_std = ta.STDDEV(close_prices, timeperiod=bb_period)
            z_score = (close_prices - rolling_mean) / rolling_std
            
            # Get current values
            current_price = close_prices[-1]
            current_bb_upper = bb_upper[-1]
            current_bb_lower = bb_lower[-1]
            current_bb_middle = bb_middle[-1]
            current_adx = adx_values[-1]
            current_zscore = z_score[-1]
            prev_price = close_prices[-2]
            prev_bb_upper = bb_upper[-2]
            prev_bb_lower = bb_lower[-2]
            
            # Check for NaN values
            if (np.isnan(current_bb_upper) or np.isnan(current_bb_lower) or 
                np.isnan(current_adx) or np.isnan(current_zscore)):
                return None
            
            action = None
            confidence = 0.0
            
            # Mean reversion conditions
            # Buy signal: Price was below lower BB, now moving back up
            if (prev_price <= prev_bb_lower and 
                current_price > current_bb_lower and 
                current_zscore < -zscore_threshold and
                current_adx < adx_threshold):  # Low ADX = ranging market good for mean reversion
                
                action = "BUY"
                
                # Calculate confidence based on oversold level and mean reversion strength
                oversold_strength = min(abs(current_zscore) / zscore_threshold, 1.0)
                mean_reversion_strength = (current_price - current_bb_lower) / (current_bb_middle - current_bb_lower)
                adx_factor = max(0, (adx_threshold - current_adx) / adx_threshold)
                confidence = min(0.6 + oversold_strength * 0.2 + mean_reversion_strength * 0.1 + adx_factor * 0.1, 1.0)
            
            # Sell signal: Price was above upper BB, now moving back down
            elif (prev_price >= prev_bb_upper and 
                  current_price < current_bb_upper and 
                  current_zscore > zscore_threshold and
                  current_adx < adx_threshold):
                
                action = "SELL"
                
                # Calculate confidence
                overbought_strength = min(abs(current_zscore) / zscore_threshold, 1.0)
                mean_reversion_strength = (current_bb_upper - current_price) / (current_bb_upper - current_bb_middle)
                adx_factor = max(0, (adx_threshold - current_adx) / adx_threshold)
                confidence = min(0.6 + overbought_strength * 0.2 + mean_reversion_strength * 0.1 + adx_factor * 0.1, 1.0)
            
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
                    'bb_upper': round(current_bb_upper, 5),
                    'bb_middle': round(current_bb_middle, 5),
                    'bb_lower': round(current_bb_lower, 5),
                    'zscore': round(current_zscore, 2),
                    'adx': round(current_adx, 2)
                }
            }
            
            logger.debug(f"Mean Reversion BB signal generated: {signal}")
            return signal
            
        except Exception as e:
            logger.error(f"Error generating Mean Reversion BB signal: {e}")
            return None
