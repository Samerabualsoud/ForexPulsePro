"""
Strategy Configuration Form Component
"""
import streamlit as st
from typing import Dict, Any, Optional

def render_strategy_form(
    strategy_name: str,
    current_config: Dict[str, Any],
    strategy_id: int,
    enabled: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Render strategy configuration form
    
    Args:
        strategy_name: Name of the strategy
        current_config: Current configuration dictionary
        strategy_id: Strategy database ID
        enabled: Whether strategy is currently enabled
    
    Returns:
        New configuration dictionary if form is submitted, None otherwise
    """
    
    form_key = f"strategy_form_{strategy_id}"
    
    with st.form(form_key):
        st.subheader(f"⚙️ {strategy_name.upper()} Configuration")
        
        # Strategy enabled toggle
        new_enabled = st.checkbox(
            "Strategy Enabled",
            value=enabled,
            key=f"enabled_{strategy_id}",
            help="Enable or disable this strategy for signal generation"
        )
        
        # Strategy-specific configuration
        if strategy_name == 'ema_rsi':
            new_config = render_ema_rsi_config(current_config, strategy_id)
        elif strategy_name == 'donchian_atr':
            new_config = render_donchian_atr_config(current_config, strategy_id)
        elif strategy_name == 'meanrev_bb':
            new_config = render_meanrev_bb_config(current_config, strategy_id)
        else:
            st.error(f"Unknown strategy: {strategy_name}")
            return None
        
        # Submit button
        submitted = st.form_submit_button(
            "💾 Save Configuration",
            use_container_width=True
        )
        
        if submitted:
            return {
                'enabled': new_enabled,
                'config': new_config
            }
    
    return None

def render_ema_rsi_config(config: Dict[str, Any], strategy_id: int) -> Dict[str, Any]:
    """Render EMA + RSI strategy configuration"""
    
    st.markdown("### 📈 EMA + RSI Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**EMA Settings**")
        ema_fast = st.number_input(
            "EMA Fast Period",
            min_value=5, max_value=50,
            value=config.get('ema_fast', 12),
            key=f"ema_fast_{strategy_id}",
            help="Fast EMA period for crossover signals"
        )
        
        ema_slow = st.number_input(
            "EMA Slow Period",
            min_value=10, max_value=100,
            value=config.get('ema_slow', 26),
            key=f"ema_slow_{strategy_id}",
            help="Slow EMA period for crossover signals"
        )
        
        st.markdown("**RSI Settings**")
        rsi_period = st.number_input(
            "RSI Period",
            min_value=5, max_value=30,
            value=config.get('rsi_period', 14),
            key=f"rsi_period_{strategy_id}",
            help="RSI calculation period"
        )
        
        rsi_threshold = st.number_input(
            "RSI Threshold",
            min_value=30, max_value=70,
            value=config.get('rsi_buy_threshold', 50),
            key=f"rsi_thresh_{strategy_id}",
            help="RSI level for signal confirmation"
        )
    
    with col2:
        st.markdown("**Risk Management**")
        sl_mode = st.selectbox(
            "Stop Loss Mode",
            options=['atr', 'pips'],
            index=0 if config.get('sl_mode') == 'atr' else 1,
            key=f"sl_mode_{strategy_id}",
            help="Method for calculating stop loss"
        )
        
        if sl_mode == 'atr':
            sl_multiplier = st.slider(
                "SL ATR Multiplier",
                min_value=1.0, max_value=5.0,
                value=config.get('sl_multiplier', 2.0),
                step=0.1,
                key=f"sl_mult_{strategy_id}",
                help="ATR multiplier for stop loss distance"
            )
            
            tp_multiplier = st.slider(
                "TP ATR Multiplier",
                min_value=1.0, max_value=8.0,
                value=config.get('tp_multiplier', 3.0),
                step=0.1,
                key=f"tp_mult_{strategy_id}",
                help="ATR multiplier for take profit distance"
            )
        else:
            sl_pips = st.number_input(
                "SL Pips",
                min_value=5, max_value=100,
                value=config.get('sl_pips', 20),
                key=f"sl_pips_{strategy_id}",
                help="Stop loss distance in pips"
            )
            
            tp_pips = st.number_input(
                "TP Pips",
                min_value=10, max_value=200,
                value=config.get('tp_pips', 40),
                key=f"tp_pips_{strategy_id}",
                help="Take profit distance in pips"
            )
        
        st.markdown("**Signal Quality**")
        min_confidence = st.slider(
            "Minimum Confidence",
            min_value=0.0, max_value=1.0,
            value=config.get('min_confidence', 0.6),
            step=0.05,
            key=f"min_conf_{strategy_id}",
            help="Minimum confidence level for signal generation"
        )
        
        expiry_bars = st.number_input(
            "Signal Expiry (minutes)",
            min_value=15, max_value=240,
            value=config.get('expiry_bars', 60),
            key=f"expiry_{strategy_id}",
            help="Signal validity period in minutes"
        )
    
    # Build configuration
    new_config = {
        'ema_fast': ema_fast,
        'ema_slow': ema_slow,
        'rsi_period': rsi_period,
        'rsi_buy_threshold': rsi_threshold,
        'rsi_sell_threshold': rsi_threshold,
        'sl_mode': sl_mode,
        'tp_mode': sl_mode,
        'min_confidence': min_confidence,
        'expiry_bars': expiry_bars
    }
    
    if sl_mode == 'atr':
        new_config.update({
            'sl_multiplier': sl_multiplier,
            'tp_multiplier': tp_multiplier
        })
    else:
        new_config.update({
            'sl_pips': sl_pips,
            'tp_pips': tp_pips
        })
    
    return new_config

def render_donchian_atr_config(config: Dict[str, Any], strategy_id: int) -> Dict[str, Any]:
    """Render Donchian + ATR strategy configuration"""
    
    st.markdown("### 📊 Donchian + ATR Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Donchian Channel**")
        donchian_period = st.number_input(
            "Donchian Period",
            min_value=10, max_value=50,
            value=config.get('donchian_period', 20),
            key=f"don_period_{strategy_id}",
            help="Period for Donchian channel calculation"
        )
        
        st.markdown("**ATR Settings**")
        atr_period = st.number_input(
            "ATR Period",
            min_value=5, max_value=30,
            value=config.get('atr_period', 14),
            key=f"atr_period_{strategy_id}",
            help="Period for ATR calculation"
        )
        
        atr_multiplier = st.slider(
            "ATR Multiplier",
            min_value=1.0, max_value=5.0,
            value=config.get('atr_multiplier', 2.0),
            step=0.1,
            key=f"atr_mult_{strategy_id}",
            help="ATR multiplier for volatility adjustment"
        )
    
    with col2:
        st.markdown("**Filters & Quality**")
        use_supertrend = st.checkbox(
            "Use SuperTrend Filter",
            value=config.get('use_supertrend', True),
            key=f"supertrend_{strategy_id}",
            help="Enable SuperTrend filter for trend confirmation"
        )
        
        min_confidence = st.slider(
            "Minimum Confidence",
            min_value=0.0, max_value=1.0,
            value=config.get('min_confidence', 0.65),
            step=0.05,
            key=f"min_conf_don_{strategy_id}",
            help="Minimum confidence level for signal generation"
        )
        
        expiry_bars = st.number_input(
            "Signal Expiry (minutes)",
            min_value=15, max_value=180,
            value=config.get('expiry_bars', 45),
            key=f"expiry_don_{strategy_id}",
            help="Signal validity period in minutes"
        )
        
        st.markdown("**Risk Management**")
        sl_multiplier = st.slider(
            "SL ATR Multiplier",
            min_value=1.0, max_value=5.0,
            value=config.get('sl_multiplier', 2.0),
            step=0.1,
            key=f"sl_mult_don_{strategy_id}",
            help="ATR multiplier for stop loss"
        )
        
        tp_multiplier = st.slider(
            "TP ATR Multiplier",
            min_value=1.5, max_value=8.0,
            value=config.get('tp_multiplier', 3.0),
            step=0.1,
            key=f"tp_mult_don_{strategy_id}",
            help="ATR multiplier for take profit"
        )
    
    return {
        'donchian_period': donchian_period,
        'atr_period': atr_period,
        'atr_multiplier': atr_multiplier,
        'use_supertrend': use_supertrend,
        'sl_mode': 'atr',
        'tp_mode': 'atr',
        'sl_multiplier': sl_multiplier,
        'tp_multiplier': tp_multiplier,
        'min_confidence': min_confidence,
        'expiry_bars': expiry_bars
    }

def render_meanrev_bb_config(config: Dict[str, Any], strategy_id: int) -> Dict[str, Any]:
    """Render Mean Reversion + Bollinger Bands strategy configuration"""
    
    st.markdown("### 🔄 Mean Reversion + BB Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Bollinger Bands**")
        bb_period = st.number_input(
            "BB Period",
            min_value=10, max_value=50,
            value=config.get('bb_period', 20),
            key=f"bb_period_{strategy_id}",
            help="Period for Bollinger Bands calculation"
        )
        
        bb_std = st.slider(
            "BB Standard Deviations",
            min_value=1.0, max_value=3.0,
            value=config.get('bb_std', 2.0),
            step=0.1,
            key=f"bb_std_{strategy_id}",
            help="Standard deviation multiplier for BB"
        )
        
        zscore_threshold = st.slider(
            "Z-Score Threshold",
            min_value=1.0, max_value=3.0,
            value=config.get('zscore_threshold', 2.0),
            step=0.1,
            key=f"zscore_{strategy_id}",
            help="Z-score threshold for mean reversion signals"
        )
    
    with col2:
        st.markdown("**ADX Filter**")
        adx_period = st.number_input(
            "ADX Period",
            min_value=10, max_value=30,
            value=config.get('adx_period', 14),
            key=f"adx_period_{strategy_id}",
            help="Period for ADX calculation"
        )
        
        adx_threshold = st.number_input(
            "ADX Threshold",
            min_value=15, max_value=40,
            value=config.get('adx_threshold', 25),
            key=f"adx_thresh_{strategy_id}",
            help="Maximum ADX for mean reversion (lower = more ranging)"
        )
        
        st.markdown("**Risk Management**")
        sl_pips = st.number_input(
            "SL Pips",
            min_value=10, max_value=50,
            value=config.get('sl_pips', 20),
            key=f"sl_pips_bb_{strategy_id}",
            help="Stop loss distance in pips"
        )
        
        tp_pips = st.number_input(
            "TP Pips",
            min_value=15, max_value=100,
            value=config.get('tp_pips', 40),
            key=f"tp_pips_bb_{strategy_id}",
            help="Take profit distance in pips"
        )
        
        min_confidence = st.slider(
            "Minimum Confidence",
            min_value=0.0, max_value=1.0,
            value=config.get('min_confidence', 0.7),
            step=0.05,
            key=f"min_conf_bb_{strategy_id}",
            help="Minimum confidence level for signal generation"
        )
    
    return {
        'bb_period': bb_period,
        'bb_std': bb_std,
        'adx_period': adx_period,
        'adx_threshold': adx_threshold,
        'zscore_threshold': zscore_threshold,
        'sl_mode': 'pips',
        'tp_mode': 'pips',
        'sl_pips': sl_pips,
        'tp_pips': tp_pips,
        'min_confidence': min_confidence,
        'expiry_bars': 30
    }

def render_strategy_info(strategy_name: str) -> None:
    """
    Render strategy information and help text
    
    Args:
        strategy_name: Name of the strategy to show info for
    """
    
    strategy_info = {
        'ema_rsi': {
            'description': 'EMA crossover with RSI confirmation',
            'logic': [
                'Buy when EMA fast crosses above EMA slow AND RSI > threshold',
                'Sell when EMA fast crosses below EMA slow AND RSI < threshold',
                'Confidence based on RSI strength and EMA separation'
            ],
            'best_for': ['Trending markets', 'Clear directional moves', 'Medium-term trades'],
            'parameters': {
                'EMA Fast': 'Shorter period EMA (default: 12)',
                'EMA Slow': 'Longer period EMA (default: 26)',
                'RSI Period': 'RSI calculation period (default: 14)',
                'RSI Threshold': 'Confirmation level (default: 50)'
            }
        },
        'donchian_atr': {
            'description': 'Donchian channel breakout with ATR volatility filter',
            'logic': [
                'Buy on breakout above Donchian upper channel',
                'Sell on breakout below Donchian lower channel',
                'ATR-based stop loss and take profit levels',
                'Optional SuperTrend filter for trend confirmation'
            ],
            'best_for': ['Breakout trading', 'Strong trending moves', 'Higher volatility periods'],
            'parameters': {
                'Donchian Period': 'Lookback period for channel (default: 20)',
                'ATR Period': 'ATR calculation period (default: 14)',
                'SuperTrend': 'Additional trend filter (recommended: enabled)'
            }
        },
        'meanrev_bb': {
            'description': 'Mean reversion using Bollinger Bands with ADX filter',
            'logic': [
                'Buy when price bounces off lower Bollinger Band',
                'Sell when price bounces off upper Bollinger Band',
                'ADX filter to avoid trending markets (low ADX preferred)',
                'Z-score based confidence calculation'
            ],
            'best_for': ['Ranging markets', 'Counter-trend trading', 'Low volatility periods'],
            'parameters': {
                'BB Period': 'Bollinger Bands period (default: 20)',
                'BB Std Dev': 'Standard deviation multiplier (default: 2.0)',
                'ADX Threshold': 'Maximum ADX for signals (default: 25)'
            }
        }
    }
    
    if strategy_name not in strategy_info:
        return
    
    info = strategy_info[strategy_name]
    
    with st.expander(f"ℹ️ {strategy_name.upper()} Strategy Information"):
        st.markdown(f"**Description:** {info['description']}")
        
        st.markdown("**Logic:**")
        for logic_point in info['logic']:
            st.markdown(f"• {logic_point}")
        
        st.markdown("**Best For:**")
        for use_case in info['best_for']:
            st.markdown(f"• {use_case}")
        
        st.markdown("**Key Parameters:**")
        for param, desc in info['parameters'].items():
            st.markdown(f"• **{param}**: {desc}")

def validate_strategy_config(strategy_name: str, config: Dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate strategy configuration
    
    Args:
        strategy_name: Name of the strategy
        config: Configuration dictionary to validate
    
    Returns:
        Tuple of (is_valid, error_messages)
    """
    
    errors = []
    
    # Common validations
    if config.get('min_confidence', 0) < 0 or config.get('min_confidence', 0) > 1:
        errors.append("Minimum confidence must be between 0.0 and 1.0")
    
    if config.get('expiry_bars', 0) < 5:
        errors.append("Signal expiry must be at least 5 minutes")
    
    # Strategy-specific validations
    if strategy_name == 'ema_rsi':
        if config.get('ema_fast', 0) >= config.get('ema_slow', 0):
            errors.append("EMA fast period must be less than EMA slow period")
        
        if config.get('rsi_period', 0) < 5:
            errors.append("RSI period must be at least 5")
    
    elif strategy_name == 'donchian_atr':
        if config.get('donchian_period', 0) < 10:
            errors.append("Donchian period must be at least 10")
        
        if config.get('atr_period', 0) < 5:
            errors.append("ATR period must be at least 5")
    
    elif strategy_name == 'meanrev_bb':
        if config.get('bb_period', 0) < 10:
            errors.append("Bollinger Bands period must be at least 10")
        
        if config.get('bb_std', 0) < 1.0:
            errors.append("BB standard deviation must be at least 1.0")
        
        if config.get('adx_threshold', 0) < 10:
            errors.append("ADX threshold must be at least 10")
    
    # Risk management validations
    if config.get('sl_mode') == 'atr':
        if config.get('sl_multiplier', 0) <= 0:
            errors.append("SL ATR multiplier must be positive")
        if config.get('tp_multiplier', 0) <= config.get('sl_multiplier', 0):
            errors.append("TP multiplier should be greater than SL multiplier")
    
    elif config.get('sl_mode') == 'pips':
        if config.get('sl_pips', 0) <= 0:
            errors.append("SL pips must be positive")
        if config.get('tp_pips', 0) <= config.get('sl_pips', 0):
            errors.append("TP pips should be greater than SL pips")
    
    return len(errors) == 0, errors
