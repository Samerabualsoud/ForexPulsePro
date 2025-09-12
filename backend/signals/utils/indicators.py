"""
Comprehensive Technical Indicators Library
All essential trading indicators for advanced analysis
"""
import pandas as pd
import numpy as np
import talib as ta
from typing import Dict, Any, Tuple, Optional

def calculate_all_indicators(data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
    """Calculate comprehensive set of technical indicators"""
    try:
        high_prices = data['high'].values
        low_prices = data['low'].values
        close_prices = data['close'].values
        open_prices = data['open'].values
        volume = data.get('volume', pd.Series(np.ones(len(data)))).values
        
        indicators = {}
        
        # === TREND INDICATORS ===
        indicators['sma_10'] = ta.SMA(close_prices, timeperiod=10)
        indicators['sma_20'] = ta.SMA(close_prices, timeperiod=20)
        indicators['sma_50'] = ta.SMA(close_prices, timeperiod=50)
        indicators['sma_200'] = ta.SMA(close_prices, timeperiod=200)
        
        indicators['ema_12'] = ta.EMA(close_prices, timeperiod=12)
        indicators['ema_26'] = ta.EMA(close_prices, timeperiod=26)
        indicators['ema_50'] = ta.EMA(close_prices, timeperiod=50)
        
        # MACD family
        macd, macd_signal, macd_hist = ta.MACD(close_prices)
        indicators['macd'] = macd
        indicators['macd_signal'] = macd_signal
        indicators['macd_histogram'] = macd_hist
        
        # Parabolic SAR
        indicators['sar'] = ta.SAR(high_prices, low_prices)
        
        # === MOMENTUM INDICATORS ===
        indicators['rsi'] = ta.RSI(close_prices, timeperiod=14)
        indicators['rsi_2'] = ta.RSI(close_prices, timeperiod=2)  # Short-term RSI
        
        # Stochastic Oscillator
        stoch_k, stoch_d = ta.STOCH(high_prices, low_prices, close_prices)
        indicators['stoch_k'] = stoch_k
        indicators['stoch_d'] = stoch_d
        
        # Stochastic RSI
        stochrsi_k, stochrsi_d = ta.STOCHRSI(close_prices)
        indicators['stochrsi_k'] = stochrsi_k
        indicators['stochrsi_d'] = stochrsi_d
        
        # Williams %R
        indicators['williams_r'] = ta.WILLR(high_prices, low_prices, close_prices)
        
        # Momentum
        indicators['mom'] = ta.MOM(close_prices, timeperiod=10)
        
        # Rate of Change
        indicators['roc'] = ta.ROC(close_prices, timeperiod=10)
        
        # Commodity Channel Index
        indicators['cci'] = ta.CCI(high_prices, low_prices, close_prices, timeperiod=14)
        
        # === VOLATILITY INDICATORS ===
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = ta.BBANDS(close_prices)
        indicators['bb_upper'] = bb_upper
        indicators['bb_middle'] = bb_middle
        indicators['bb_lower'] = bb_lower
        indicators['bb_width'] = (bb_upper - bb_lower) / bb_middle * 100
        
        # Average True Range
        indicators['atr'] = ta.ATR(high_prices, low_prices, close_prices)
        
        # Donchian Channels
        indicators['donchian_upper'] = ta.MAX(high_prices, timeperiod=20)
        indicators['donchian_lower'] = ta.MIN(low_prices, timeperiod=20)
        indicators['donchian_middle'] = (indicators['donchian_upper'] + indicators['donchian_lower']) / 2
        
        # Standard Deviation
        indicators['stddev'] = ta.STDDEV(close_prices, timeperiod=20)
        
        # === VOLUME INDICATORS ===
        # On-Balance Volume
        indicators['obv'] = ta.OBV(close_prices, volume)
        
        # Volume SMA
        indicators['volume_sma'] = ta.SMA(volume, timeperiod=20)
        
        # Accumulation/Distribution Line
        indicators['ad'] = ta.AD(high_prices, low_prices, close_prices, volume)
        
        # Chaikin A/D Oscillator
        indicators['adosc'] = ta.ADOSC(high_prices, low_prices, close_prices, volume)
        
        # === SUPPORT/RESISTANCE ===
        # Pivot Points (Traditional)
        indicators['pivot'] = (high_prices[-1] + low_prices[-1] + close_prices[-1]) / 3
        indicators['r1'] = 2 * indicators['pivot'] - low_prices[-1]  # Resistance 1
        indicators['s1'] = 2 * indicators['pivot'] - high_prices[-1]  # Support 1
        indicators['r2'] = indicators['pivot'] + (high_prices[-1] - low_prices[-1])
        indicators['s2'] = indicators['pivot'] - (high_prices[-1] - low_prices[-1])
        
        # === CANDLESTICK PATTERNS ===
        indicators['doji'] = ta.CDLDOJI(open_prices, high_prices, low_prices, close_prices)
        indicators['hammer'] = ta.CDLHAMMER(open_prices, high_prices, low_prices, close_prices)
        indicators['engulfing'] = ta.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)
        indicators['morning_star'] = ta.CDLMORNINGSTAR(open_prices, high_prices, low_prices, close_prices)
        indicators['evening_star'] = ta.CDLEVENINGSTAR(open_prices, high_prices, low_prices, close_prices)
        
        # === ICHIMOKU CLOUD ===
        # Tenkan-sen (Conversion Line)
        period9_high = ta.MAX(high_prices, timeperiod=9)
        period9_low = ta.MIN(low_prices, timeperiod=9)
        indicators['ichimoku_tenkan'] = (period9_high + period9_low) / 2
        
        # Kijun-sen (Base Line)
        period26_high = ta.MAX(high_prices, timeperiod=26)
        period26_low = ta.MIN(low_prices, timeperiod=26)
        indicators['ichimoku_kijun'] = (period26_high + period26_low) / 2
        
        # Senkou Span A (Leading Span A)
        indicators['ichimoku_senkou_a'] = (indicators['ichimoku_tenkan'] + indicators['ichimoku_kijun']) / 2
        
        # Senkou Span B (Leading Span B)
        period52_high = ta.MAX(high_prices, timeperiod=52)
        period52_low = ta.MIN(low_prices, timeperiod=52)
        indicators['ichimoku_senkou_b'] = (period52_high + period52_low) / 2
        
        # === FIBONACCI LEVELS ===
        recent_high = np.max(high_prices[-20:])
        recent_low = np.min(low_prices[-20:])
        fib_diff = recent_high - recent_low
        
        indicators['fib_23_6'] = recent_low + fib_diff * 0.236
        indicators['fib_38_2'] = recent_low + fib_diff * 0.382
        indicators['fib_50_0'] = recent_low + fib_diff * 0.500
        indicators['fib_61_8'] = recent_low + fib_diff * 0.618
        indicators['fib_78_6'] = recent_low + fib_diff * 0.786
        
        # === MARKET SENTIMENT ===
        # Fear & Greed Index (simplified)
        rsi_val = indicators['rsi'][-1] if not np.isnan(indicators['rsi'][-1]) else 50
        bb_position = (close_prices[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1]) * 100
        
        if not np.isnan(bb_position):
            sentiment_score = (rsi_val + bb_position) / 2
            if sentiment_score > 70:
                indicators['market_sentiment'] = 'GREED'
            elif sentiment_score < 30:
                indicators['market_sentiment'] = 'FEAR'
            else:
                indicators['market_sentiment'] = 'NEUTRAL'
        else:
            indicators['market_sentiment'] = 'NEUTRAL'
            
        return indicators
        
    except Exception as e:
        print(f"Error calculating indicators for {symbol}: {e}")
        return {}

def get_signal_strength(indicators: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    """Analyze overall signal strength from all indicators"""
    try:
        bullish_signals = 0
        bearish_signals = 0
        total_signals = 0
        
        # Price vs Moving Averages
        current_price = indicators.get('close', 0)
        if current_price > 0:
            mas = ['sma_20', 'sma_50', 'ema_12', 'ema_26']
            for ma in mas:
                if ma in indicators and not np.isnan(indicators[ma][-1]):
                    total_signals += 1
                    if current_price > indicators[ma][-1]:
                        bullish_signals += 1
                    else:
                        bearish_signals += 1
        
        # RSI Analysis
        if 'rsi' in indicators and not np.isnan(indicators['rsi'][-1]):
            rsi_val = indicators['rsi'][-1]
            total_signals += 1
            if rsi_val < 30:  # Oversold - potentially bullish
                bullish_signals += 1
            elif rsi_val > 70:  # Overbought - potentially bearish
                bearish_signals += 1
        
        # MACD Analysis
        if ('macd' in indicators and 'macd_signal' in indicators and
            not np.isnan(indicators['macd'][-1]) and not np.isnan(indicators['macd_signal'][-1])):
            total_signals += 1
            if indicators['macd'][-1] > indicators['macd_signal'][-1]:
                bullish_signals += 1
            else:
                bearish_signals += 1
                
        # Stochastic Analysis
        if ('stoch_k' in indicators and 'stoch_d' in indicators and
            not np.isnan(indicators['stoch_k'][-1]) and not np.isnan(indicators['stoch_d'][-1])):
            total_signals += 1
            if indicators['stoch_k'][-1] > indicators['stoch_d'][-1]:
                bullish_signals += 1
            else:
                bearish_signals += 1
        
        # Calculate overall sentiment
        if total_signals > 0:
            bullish_ratio = bullish_signals / total_signals
            bearish_ratio = bearish_signals / total_signals
            
            if bullish_ratio > 0.6:
                overall_sentiment = 'BULLISH'
                confidence = bullish_ratio
            elif bearish_ratio > 0.6:
                overall_sentiment = 'BEARISH' 
                confidence = bearish_ratio
            else:
                overall_sentiment = 'NEUTRAL'
                confidence = 0.5
        else:
            overall_sentiment = 'NEUTRAL'
            confidence = 0.5
            
        return {
            'overall_sentiment': overall_sentiment,
            'confidence': round(confidence, 2),
            'bullish_signals': bullish_signals,
            'bearish_signals': bearish_signals,
            'total_signals': total_signals,
            'market_sentiment': indicators.get('market_sentiment', 'NEUTRAL')
        }
        
    except Exception as e:
        print(f"Error calculating signal strength for {symbol}: {e}")
        return {
            'overall_sentiment': 'NEUTRAL',
            'confidence': 0.5,
            'bullish_signals': 0,
            'bearish_signals': 0,
            'total_signals': 0,
            'market_sentiment': 'NEUTRAL'
        }