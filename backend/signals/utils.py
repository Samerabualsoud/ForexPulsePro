"""
Signal Utility Functions
"""
import pandas as pd
import numpy as np
import talib as ta
from typing import Dict, Any, Tuple, Optional

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
    config: Dict[str, Any]
) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate Stop Loss and Take Profit levels
    
    Args:
        price: Entry price
        action: BUY or SELL
        data: OHLC data
        config: Strategy configuration
        
    Returns:
        Tuple of (stop_loss, take_profit)
    """
    sl_mode = config.get('sl_mode', 'atr')
    tp_mode = config.get('tp_mode', 'atr')
    
    sl = None
    tp = None
    
    # Calculate Stop Loss
    if sl_mode == 'pips':
        sl_pips = config.get('sl_pips', 20)
        pip_value = get_pip_value(data['close'].iloc[-1])
        if action == 'BUY':
            sl = price - (sl_pips * pip_value)
        else:
            sl = price + (sl_pips * pip_value)
    
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
        pip_value = get_pip_value(data['close'].iloc[-1])
        if action == 'BUY':
            tp = price + (tp_pips * pip_value)
        else:
            tp = price - (tp_pips * pip_value)
    
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

def get_pip_value(price: float) -> float:
    """
    Get pip value based on price level
    Most forex pairs have 5 decimal places, JPY pairs have 3
    """
    if price > 50:  # Likely a JPY pair
        return 0.01
    else:
        return 0.0001

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
    Calculate position size based on risk management
    
    Args:
        account_balance: Account balance in base currency
        risk_percentage: Risk percentage (e.g., 0.02 for 2%)
        entry_price: Entry price
        stop_loss: Stop loss price
        symbol: Currency pair symbol
        
    Returns:
        Position size in lots
    """
    risk_amount = account_balance * risk_percentage
    
    # Calculate pip risk
    pip_value = get_pip_value(entry_price)
    pip_risk = abs(entry_price - stop_loss) / pip_value
    
    # Calculate lot size (assuming $10 per pip per lot for major pairs)
    pip_value_per_lot = 10.0  # This is simplified - should be calculated based on account currency
    lot_size = risk_amount / (pip_risk * pip_value_per_lot)
    
    # Round to appropriate precision
    return round(lot_size, 2)

def format_signal_message(signal_data: Dict[str, Any]) -> str:
    """
    Format signal data into WhatsApp message template
    
    Template: {{symbol}} {{action}} @ {{price}} | SL {{sl}} | TP {{tp}} | conf {{confidence}} | {{strategy}}
    """
    symbol = signal_data.get('symbol', 'UNKNOWN')
    action = signal_data.get('action', 'FLAT')
    price = signal_data.get('price', 0.0)
    sl = signal_data.get('sl', 'N/A')
    tp = signal_data.get('tp', 'N/A')
    confidence = signal_data.get('confidence', 0.0)
    strategy = signal_data.get('strategy', 'unknown')
    
    # Format SL and TP
    sl_str = f"{sl:.5f}" if sl and sl != 'N/A' else 'N/A'
    tp_str = f"{tp:.5f}" if tp and tp != 'N/A' else 'N/A'
    
    message = f"{symbol} {action} @ {price:.5f} | SL {sl_str} | TP {tp_str} | conf {confidence:.2f} | {strategy}"
    
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
