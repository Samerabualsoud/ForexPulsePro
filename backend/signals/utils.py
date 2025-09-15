"""
Signal Utility Functions
"""
import pandas as pd
import numpy as np
import talib as ta
from typing import Dict, Any, Tuple, Optional

# Import instrument metadata system
from ..instruments.metadata import (
    instrument_db, get_instrument_metadata, get_pip_size, get_pip_value_per_lot,
    format_price, round_lot_size
)

def calculate_atr(data: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Calculate Average True Range"""
    high_prices = data['high'].values
    low_prices = data['low'].values
    close_prices = data['close'].values
    
    return ta.ATR(high_prices, low_prices, close_prices, timeperiod=period)

def calculate_sl_tp(
    price: float, 
    action: str, 
    data: pd.DataFrame, 
    config: Dict[str, Any],
    symbol: str = None
) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate Stop Loss and Take Profit levels using accurate instrument metadata
    
    Args:
        price: Entry price
        action: BUY or SELL
        data: OHLC data
        config: Strategy configuration
        symbol: Trading symbol for metadata lookup
        
    Returns:
        Tuple of (stop_loss, take_profit)
    """
    sl_mode = config.get('sl_mode', 'atr')
    tp_mode = config.get('tp_mode', 'atr')
    
    sl = None
    tp = None
    
    # Get accurate pip size using instrument metadata
    pip_size = get_pip_size(symbol) if symbol else get_pip_value_legacy(data['close'].iloc[-1])
    
    # Calculate Stop Loss
    if sl_mode == 'pips':
        sl_pips = config.get('sl_pips', 20)
        if action == 'BUY':
            sl = price - (sl_pips * pip_size)
        else:
            sl = price + (sl_pips * pip_size)
    
    elif sl_mode == 'atr':
        atr_values = calculate_atr(data)
        if not np.isnan(atr_values[-1]):
            sl_multiplier = config.get('sl_multiplier', 2.0)
            atr_sl = atr_values[-1] * sl_multiplier
            
            if action == 'BUY':
                sl = price - atr_sl
            else:
                sl = price + atr_sl
    
    # Calculate Take Profit
    if tp_mode == 'pips':
        tp_pips = config.get('tp_pips', 40)
        if action == 'BUY':
            tp = price + (tp_pips * pip_size)
        else:
            tp = price - (tp_pips * pip_size)
    
    elif tp_mode == 'atr':
        atr_values = calculate_atr(data)
        if not np.isnan(atr_values[-1]):
            tp_multiplier = config.get('tp_multiplier', 3.0)
            atr_tp = atr_values[-1] * tp_multiplier
            
            if action == 'BUY':
                tp = price + atr_tp
            else:
                tp = price - atr_tp
    
    return sl, tp

def get_pip_value_legacy(price: float) -> float:
    """
    Legacy pip value calculation - kept for backward compatibility
    Use get_pip_size() from instrument metadata instead
    """
    if price > 50:  # Likely a JPY pair
        return 0.01
    else:
        return 0.0001

def get_pip_value(symbol_or_price, symbol: str = None) -> float:
    """
    Get pip value - supports both legacy price-based and new symbol-based lookup
    
    Args:
        symbol_or_price: Either a symbol string or price float (legacy)
        symbol: Symbol string when first param is price (legacy mode)
        
    Returns:
        Pip size for the instrument
    """
    if isinstance(symbol_or_price, str):
        # New metadata-based lookup
        return get_pip_size(symbol_or_price)
    else:
        # Legacy price-based lookup with optional symbol
        if symbol:
            return get_pip_size(symbol)
        else:
            return get_pip_value_legacy(symbol_or_price)

def normalize_symbol(symbol: str) -> str:
    """Normalize symbol format"""
    return symbol.upper().replace('/', '').replace('-', '')

def calculate_position_size(
    account_balance: float,
    risk_percentage: float,
    entry_price: float,
    stop_loss: float,
    symbol: str
) -> float:
    """
    Calculate position size based on risk management using accurate instrument metadata
    
    Args:
        account_balance: Account balance in base currency
        risk_percentage: Risk percentage (e.g., 0.02 for 2%)
        entry_price: Entry price
        stop_loss: Stop loss price
        symbol: Trading symbol
        
    Returns:
        Position size in lots (properly rounded to valid increments)
    """
    risk_amount = account_balance * risk_percentage
    
    # Get accurate pip size and pip value per lot from metadata
    pip_size = get_pip_size(symbol)
    pip_value_per_lot_usd = get_pip_value_per_lot(symbol)
    
    # Calculate pip risk
    pip_risk = abs(entry_price - stop_loss) / pip_size
    
    # Calculate raw lot size
    lot_size = risk_amount / (pip_risk * pip_value_per_lot_usd)
    
    # Round to valid lot size using instrument specifications
    return round_lot_size(symbol, lot_size)

def format_signal_message(signal_data: Dict[str, Any]) -> str:
    """
    Format signal data into WhatsApp message template using proper decimal precision
    
    Template: {{symbol}} {{action}} @ {{price}} | SL {{sl}} | TP {{tp}} | conf {{confidence}} | {{strategy}}
    """
    symbol = signal_data.get('symbol', 'UNKNOWN')
    action = signal_data.get('action', 'FLAT')
    price = signal_data.get('price', 0.0)
    sl = signal_data.get('sl', 'N/A')
    tp = signal_data.get('tp', 'N/A')
    confidence = signal_data.get('confidence', 0.0)
    strategy = signal_data.get('strategy', 'unknown')
    
    # Format prices using instrument metadata for correct decimal places
    price_str = format_price(symbol, price) if symbol != 'UNKNOWN' else f"{price:.5f}"
    sl_str = format_price(symbol, sl) if sl and sl != 'N/A' and symbol != 'UNKNOWN' else ('N/A' if sl == 'N/A' else f"{sl:.5f}")
    tp_str = format_price(symbol, tp) if tp and tp != 'N/A' and symbol != 'UNKNOWN' else ('N/A' if tp == 'N/A' else f"{tp:.5f}")
    
    message = f"{symbol} {action} @ {price_str} | SL {sl_str} | TP {tp_str} | conf {confidence:.2f} | {strategy}"
    
    return message

def validate_signal_data(signal_data: Dict[str, Any]) -> bool:
    """Validate signal data completeness and correctness"""
    required_fields = ['symbol', 'action', 'price', 'confidence', 'strategy']
    
    # Check required fields
    for field in required_fields:
        if field not in signal_data:
            return False
    
    # Validate action - now includes MT5 order types
    valid_actions = ['BUY', 'SELL', 'FLAT', 'BUY LIMIT', 'SELL LIMIT', 
                     'BUY STOP', 'SELL STOP', 'BUY STOP LIMIT', 'SELL STOP LIMIT']
    if signal_data['action'] not in valid_actions:
        return False
    
    # Validate numeric fields
    try:
        float(signal_data['price'])
        float(signal_data['confidence'])
        
        if signal_data.get('sl'):
            float(signal_data['sl'])
        if signal_data.get('tp'):
            float(signal_data['tp'])
    except (ValueError, TypeError):
        return False
    
    # Validate confidence range
    confidence = float(signal_data['confidence'])
    if confidence < 0.0 or confidence > 1.0:
        return False
    
    return True


def determine_market_regime(data: pd.DataFrame, config: Dict[str, Any] = None) -> str:
    """
    Determine if market is in trending or ranging regime
    
    Args:
        data: OHLC data
        config: Configuration parameters
        
    Returns:
        'TRENDING' or 'RANGING'
    """
    try:
        if len(data) < 20:
            return 'RANGING'  # Default to ranging for insufficient data
            
        close_prices = data['close'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        
        # Calculate ADX for trend strength
        adx = ta.ADX(high_prices, low_prices, close_prices, timeperiod=14)
        adx_threshold = config.get('adx_threshold', 25) if config else 25
        
        # Calculate price movement efficiency
        price_range = abs(close_prices[-1] - close_prices[-20])
        cumulative_movement = sum(abs(close_prices[i] - close_prices[i-1]) for i in range(1, 21))
        efficiency = price_range / cumulative_movement if cumulative_movement > 0 else 0
        
        current_adx = adx[-1] if not np.isnan(adx[-1]) else 20
        
        # Trending if ADX is high and price movement is efficient
        if current_adx > adx_threshold and efficiency > 0.4:
            return 'TRENDING'
        else:
            return 'RANGING'
            
    except Exception:
        return 'RANGING'


def calculate_volatility_factor(data: pd.DataFrame, period: int = 14) -> float:
    """
    Calculate normalized volatility factor
    
    Args:
        data: OHLC data
        period: ATR calculation period
        
    Returns:
        Volatility factor (0.0 to 1.0)
    """
    try:
        atr_values = calculate_atr(data, period)
        current_atr = atr_values[-1]
        current_price = data['close'].iloc[-1]
        
        # Normalize ATR as percentage of price
        volatility_pct = (current_atr / current_price) * 100
        
        # Scale to 0-1 range (assuming max normal volatility is 3%)
        volatility_factor = min(volatility_pct / 3.0, 1.0)
        
        return volatility_factor
        
    except Exception:
        return 0.5  # Medium volatility as default


def determine_mt5_order_type(
    signal_price: float,
    current_price: float, 
    base_action: str,
    data: pd.DataFrame,
    config: Dict[str, Any] = None,
    strategy_type: str = None
) -> str:
    """
    Determine the appropriate MT5 order type based on market conditions
    
    Args:
        signal_price: Target entry price from strategy
        current_price: Current market price
        base_action: Base signal direction ('BUY' or 'SELL')
        data: OHLC price data
        config: Strategy configuration
        strategy_type: Type of strategy (breakout, mean_reversion, momentum, etc.)
        
    Returns:
        MT5 order type string
    """
    if base_action not in ['BUY', 'SELL']:
        return base_action  # Return as-is for FLAT or other actions
        
    config = config or {}
    
    # Get market conditions
    market_regime = determine_market_regime(data, config)
    volatility_factor = calculate_volatility_factor(data)
    
    # Calculate price differential
    price_diff_pct = abs(signal_price - current_price) / current_price
    
    # Threshold for immediate vs pending orders (default 0.1% = 10 pips on major pairs)
    immediate_threshold = config.get('immediate_threshold_pct', 0.001)
    
    # For very small price differences, use immediate orders
    if price_diff_pct <= immediate_threshold:
        return base_action  # BUY or SELL (market order)
    
    # Determine order type based on strategy and market conditions
    if base_action == 'BUY':
        if signal_price < current_price:
            # Buying below current price - expecting pullback
            if strategy_type == 'mean_reversion' or market_regime == 'RANGING':
                return 'BUY LIMIT'
            elif volatility_factor > 0.7:
                # High volatility - use stop limit for protection
                return 'BUY STOP LIMIT'
            else:
                return 'BUY LIMIT'
        else:
            # Buying above current price - expecting breakout/continuation
            if strategy_type in ['breakout', 'momentum'] or market_regime == 'TRENDING':
                return 'BUY STOP'
            elif volatility_factor > 0.7:
                return 'BUY STOP LIMIT'
            else:
                return 'BUY STOP'
                
    else:  # SELL
        if signal_price > current_price:
            # Selling above current price - expecting pullback  
            if strategy_type == 'mean_reversion' or market_regime == 'RANGING':
                return 'SELL LIMIT'
            elif volatility_factor > 0.7:
                return 'SELL STOP LIMIT'
            else:
                return 'SELL LIMIT'
        else:
            # Selling below current price - expecting breakout/continuation
            if strategy_type in ['breakout', 'momentum'] or market_regime == 'TRENDING':
                return 'SELL STOP'
            elif volatility_factor > 0.7:
                return 'SELL STOP LIMIT'
            else:
                return 'SELL STOP'


def adjust_signal_price_for_order_type(
    signal_price: float,
    current_price: float,
    order_type: str,
    data: pd.DataFrame,
    config: Dict[str, Any] = None
) -> float:
    """
    Adjust signal price based on order type and market conditions
    
    Args:
        signal_price: Original signal price
        current_price: Current market price
        order_type: Determined MT5 order type
        data: OHLC price data
        config: Configuration parameters
        
    Returns:
        Adjusted signal price
    """
    config = config or {}
    
    # For market orders, use current price
    if order_type in ['BUY', 'SELL']:
        return current_price
        
    # For pending orders, we might want to adjust the price slightly
    # to account for spread, slippage, or strategic positioning
    
    pip_size = get_pip_value_legacy(current_price)  # Use legacy for now
    buffer_pips = config.get('order_buffer_pips', 2)  # Small buffer
    
    if 'LIMIT' in order_type:
        # For limit orders, we can be more aggressive (closer to current price)
        if order_type.startswith('BUY'):
            # Buy limit should be below current price
            adjusted_price = min(signal_price, current_price - (buffer_pips * pip_size))
        else:
            # Sell limit should be above current price  
            adjusted_price = max(signal_price, current_price + (buffer_pips * pip_size))
    
    elif 'STOP' in order_type:
        # For stop orders, add small buffer for activation
        if order_type.startswith('BUY'):
            # Buy stop should be above current price
            adjusted_price = max(signal_price, current_price + (buffer_pips * pip_size))
        else:
            # Sell stop should be below current price
            adjusted_price = min(signal_price, current_price - (buffer_pips * pip_size))
    else:
        adjusted_price = signal_price
        
    return adjusted_price


def enhance_signal_with_mt5_order_type(
    signal_data: Dict[str, Any],
    data: pd.DataFrame,
    config: Dict[str, Any] = None,
    strategy_type: str = None
) -> Dict[str, Any]:
    """
    Enhance existing signal data with MT5 order type determination
    
    Args:
        signal_data: Original signal data with basic BUY/SELL action
        data: OHLC price data
        config: Strategy configuration
        strategy_type: Strategy type hint
        
    Returns:
        Enhanced signal data with MT5 order type
    """
    if not signal_data or signal_data.get('action') not in ['BUY', 'SELL']:
        return signal_data
        
    current_price = data['close'].iloc[-1]
    signal_price = signal_data.get('price', current_price)
    base_action = signal_data['action']
    
    # Determine MT5 order type
    mt5_order_type = determine_mt5_order_type(
        signal_price=signal_price,
        current_price=current_price,
        base_action=base_action,
        data=data,
        config=config,
        strategy_type=strategy_type
    )
    
    # Adjust price if needed
    adjusted_price = adjust_signal_price_for_order_type(
        signal_price=signal_price,
        current_price=current_price,
        order_type=mt5_order_type,
        data=data,
        config=config
    )
    
    # Update signal data
    enhanced_signal = signal_data.copy()
    enhanced_signal['action'] = mt5_order_type
    enhanced_signal['price'] = adjusted_price
    enhanced_signal['original_action'] = base_action  # Keep original for reference
    enhanced_signal['current_market_price'] = current_price
    enhanced_signal['order_type_reasoning'] = _generate_order_type_reasoning(
        mt5_order_type, signal_price, current_price, strategy_type
    )
    
    return enhanced_signal


def _generate_order_type_reasoning(
    order_type: str,
    signal_price: float, 
    current_price: float,
    strategy_type: str = None
) -> str:
    """Generate human-readable reasoning for order type selection"""
    
    price_diff = signal_price - current_price
    direction = "above" if price_diff > 0 else "below"
    
    reasoning_map = {
        'BUY': f"Immediate buy at market price",
        'SELL': f"Immediate sell at market price", 
        'BUY LIMIT': f"Buy limit {direction} market - expecting pullback to {signal_price:.5f}",
        'SELL LIMIT': f"Sell limit {direction} market - expecting bounce to {signal_price:.5f}",
        'BUY STOP': f"Buy stop {direction} market - breakout continuation at {signal_price:.5f}",
        'SELL STOP': f"Sell stop {direction} market - breakdown continuation at {signal_price:.5f}",
        'BUY STOP LIMIT': f"Buy stop limit {direction} market - protected breakout entry",
        'SELL STOP LIMIT': f"Sell stop limit {direction} market - protected breakdown entry"
    }
    
    base_reason = reasoning_map.get(order_type, f"Order type: {order_type}")
    
    if strategy_type:
        base_reason += f" [{strategy_type} strategy]"
        
    return base_reason
