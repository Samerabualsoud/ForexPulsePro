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
    
    # Validate action
    if signal_data['action'] not in ['BUY', 'SELL', 'FLAT']:
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
