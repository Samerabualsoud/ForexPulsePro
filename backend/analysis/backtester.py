"""
Advanced Backtesting System for Strategy Performance Analysis
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import structlog
from dataclasses import dataclass
from enum import Enum

logger = structlog.get_logger(__name__)

class TradeStatus(Enum):
    OPEN = "open"
    CLOSED_PROFIT = "closed_profit"
    CLOSED_LOSS = "closed_loss"
    CLOSED_BREAKEVEN = "closed_breakeven"

@dataclass
class Trade:
    """Individual trade representation"""
    symbol: str
    action: str  # BUY/SELL
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    strategy: str
    confidence: float
    
    # Results (filled when closed)
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: Optional[float] = None
    pnl_pips: Optional[float] = None
    status: TradeStatus = TradeStatus.OPEN
    duration_minutes: Optional[int] = None

@dataclass
class StrategyMetrics:
    """Performance metrics for a trading strategy"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0
    
    total_pnl: float = 0.0
    total_pips: float = 0.0
    
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    
    max_drawdown: float = 0.0
    max_consecutive_losses: int = 0
    avg_trade_duration: float = 0.0
    
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    best_trade: float = 0.0
    worst_trade: float = 0.0

class AdvancedBacktester:
    """Professional-grade backtesting engine with comprehensive analysis"""
    
    def __init__(self):
        self.trades: List[Trade] = []
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.strategy_metrics: Dict[str, StrategyMetrics] = {}
        
    def load_market_data(self, symbol: str, data: List[Dict]) -> None:
        """Load OHLC market data for backtesting"""
        try:
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()
            
            self.market_data[symbol] = df
            logger.info(f"Loaded {len(df)} bars for {symbol}")
            
        except Exception as e:
            logger.error(f"Error loading market data for {symbol}: {e}")
    
    def add_trade(self, trade: Trade) -> None:
        """Add trade to backtest"""
        self.trades.append(trade)
    
    def simulate_trade_execution(self, trade: Trade, market_data: pd.DataFrame) -> Trade:
        """
        Simulate trade execution with realistic price fills
        
        Args:
            trade: Trade to simulate
            market_data: OHLC data for the symbol
            
        Returns:
            Completed trade with results
        """
        try:
            # Find entry bar
            entry_bars = market_data[market_data.index >= trade.entry_time]
            if entry_bars.empty:
                logger.warning(f"No market data available for trade entry at {trade.entry_time}")
                return trade
            
            entry_bar = entry_bars.iloc[0]
            actual_entry_price = trade.entry_price
            
            # Check if entry price is realistic (within bar range)
            if not (entry_bar['low'] <= actual_entry_price <= entry_bar['high']):
                # Adjust to nearest realistic price
                if trade.action == "BUY":
                    actual_entry_price = min(actual_entry_price, entry_bar['high'])
                else:  # SELL
                    actual_entry_price = max(actual_entry_price, entry_bar['low'])
            
            trade.entry_price = actual_entry_price
            
            # Simulate trade progression bar by bar
            remaining_bars = market_data[market_data.index > trade.entry_time]
            
            for timestamp, bar in remaining_bars.iterrows():
                # Check for stop loss hit
                if trade.action == "BUY":
                    if bar['low'] <= trade.stop_loss:
                        # Stop loss hit
                        trade.exit_price = trade.stop_loss
                        trade.exit_time = timestamp
                        trade.pnl = trade.stop_loss - trade.entry_price
                        trade.status = TradeStatus.CLOSED_LOSS
                        break
                    elif bar['high'] >= trade.take_profit:
                        # Take profit hit
                        trade.exit_price = trade.take_profit
                        trade.exit_time = timestamp
                        trade.pnl = trade.take_profit - trade.entry_price
                        trade.status = TradeStatus.CLOSED_PROFIT
                        break
                else:  # SELL
                    if bar['high'] >= trade.stop_loss:
                        # Stop loss hit
                        trade.exit_price = trade.stop_loss
                        trade.exit_time = timestamp
                        trade.pnl = trade.entry_price - trade.stop_loss
                        trade.status = TradeStatus.CLOSED_LOSS
                        break
                    elif bar['low'] <= trade.take_profit:
                        # Take profit hit
                        trade.exit_price = trade.take_profit
                        trade.exit_time = timestamp
                        trade.pnl = trade.entry_price - trade.take_profit
                        trade.status = TradeStatus.CLOSED_PROFIT
                        break
            
            # Calculate additional metrics
            if trade.exit_time:
                trade.duration_minutes = int((trade.exit_time - trade.entry_time).total_seconds() / 60)
                
                # Convert PnL to pips (assuming 4-decimal pair, adjust for JPY pairs)
                pip_multiplier = 10000 if 'JPY' not in trade.symbol else 100
                trade.pnl_pips = trade.pnl * pip_multiplier
                
                # Determine final status based on PnL
                if abs(trade.pnl) < 0.0001:  # Essentially breakeven
                    trade.status = TradeStatus.CLOSED_BREAKEVEN
                elif trade.pnl > 0:
                    trade.status = TradeStatus.CLOSED_PROFIT
                else:
                    trade.status = TradeStatus.CLOSED_LOSS
            
            return trade
            
        except Exception as e:
            logger.error(f"Error simulating trade execution: {e}")
            return trade
    
    def run_backtest(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Run comprehensive backtest simulation
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Backtest results and metrics
        """
        logger.info(f"Starting backtest with {len(self.trades)} trades")
        
        # Filter trades by date range if provided
        filtered_trades = self.trades
        if start_date or end_date:
            filtered_trades = [
                trade for trade in self.trades
                if (not start_date or trade.entry_time >= start_date) and
                   (not end_date or trade.entry_time <= end_date)
            ]
        
        # Simulate each trade
        simulated_trades = []
        for trade in filtered_trades:
            if trade.symbol in self.market_data:
                simulated_trade = self.simulate_trade_execution(trade, self.market_data[trade.symbol])
                simulated_trades.append(simulated_trade)
            else:
                logger.warning(f"No market data for {trade.symbol}, skipping trade")
        
        # Calculate performance metrics
        overall_metrics = self.calculate_strategy_metrics(simulated_trades)
        
        # Calculate metrics by strategy
        strategy_breakdown = {}
        for strategy in set(trade.strategy for trade in simulated_trades):
            strategy_trades = [t for t in simulated_trades if t.strategy == strategy]
            strategy_breakdown[strategy] = self.calculate_strategy_metrics(strategy_trades)
        
        # Calculate additional analytics
        equity_curve = self.calculate_equity_curve(simulated_trades)
        monthly_returns = self.calculate_monthly_returns(simulated_trades)
        drawdown_analysis = self.calculate_drawdown_analysis(equity_curve)
        
        results = {
            "overall_metrics": overall_metrics,
            "strategy_breakdown": strategy_breakdown,
            "equity_curve": equity_curve,
            "monthly_returns": monthly_returns,
            "drawdown_analysis": drawdown_analysis,
            "trade_details": simulated_trades,
            "backtest_period": {
                "start": start_date or (min(t.entry_time for t in simulated_trades) if simulated_trades else None),
                "end": end_date or (max(t.entry_time for t in simulated_trades) if simulated_trades else None)
            }
        }
        
        logger.info(f"Backtest completed: {overall_metrics.total_trades} trades, {overall_metrics.win_rate:.1%} win rate")
        return results
    
    def calculate_strategy_metrics(self, trades: List[Trade]) -> StrategyMetrics:
        """Calculate comprehensive performance metrics for trades"""
        if not trades:
            return StrategyMetrics()
        
        completed_trades = [t for t in trades if t.status != TradeStatus.OPEN]
        if not completed_trades:
            return StrategyMetrics()
        
        metrics = StrategyMetrics()
        metrics.total_trades = len(completed_trades)
        
        # Basic statistics
        profits = [t.pnl for t in completed_trades if t.pnl and t.pnl > 0]
        losses = [abs(t.pnl) for t in completed_trades if t.pnl and t.pnl < 0]
        
        metrics.winning_trades = len(profits)
        metrics.losing_trades = len(losses)
        metrics.breakeven_trades = metrics.total_trades - metrics.winning_trades - metrics.losing_trades
        
        metrics.total_pnl = sum(t.pnl or 0 for t in completed_trades)
        metrics.total_pips = sum(t.pnl_pips or 0 for t in completed_trades)
        
        # Performance ratios
        if metrics.total_trades > 0:
            metrics.win_rate = metrics.winning_trades / metrics.total_trades
        
        if profits:
            metrics.avg_win = np.mean(profits)
            metrics.best_trade = max(t.pnl for t in completed_trades if t.pnl)
        
        if losses:
            metrics.avg_loss = np.mean(losses)
            metrics.worst_trade = min(t.pnl for t in completed_trades if t.pnl)
        
        if metrics.avg_loss > 0:
            metrics.profit_factor = abs(metrics.avg_win * metrics.winning_trades) / (metrics.avg_loss * metrics.losing_trades)
        
        # Risk metrics
        if completed_trades:
            durations = [t.duration_minutes for t in completed_trades if t.duration_minutes]
            if durations:
                metrics.avg_trade_duration = np.mean(durations)
        
        # Calculate drawdown
        equity_curve = self.calculate_equity_curve(completed_trades)
        if equity_curve:
            running_max = 0
            max_dd = 0
            for equity in equity_curve:
                running_max = max(running_max, equity)
                drawdown = (running_max - equity) / running_max if running_max > 0 else 0
                max_dd = max(max_dd, drawdown)
            metrics.max_drawdown = max_dd
        
        # Calculate consecutive losses
        consecutive_losses = 0
        max_consecutive = 0
        for trade in completed_trades:
            if trade.pnl and trade.pnl < 0:
                consecutive_losses += 1
                max_consecutive = max(max_consecutive, consecutive_losses)
            else:
                consecutive_losses = 0
        metrics.max_consecutive_losses = max_consecutive
        
        return metrics
    
    def calculate_equity_curve(self, trades: List[Trade]) -> List[float]:
        """Calculate equity curve progression"""
        equity_curve = [0.0]  # Start with 0
        running_pnl = 0.0
        
        for trade in sorted(trades, key=lambda t: t.entry_time):
            if trade.pnl is not None:
                running_pnl += trade.pnl
                equity_curve.append(running_pnl)
        
        return equity_curve
    
    def calculate_monthly_returns(self, trades: List[Trade]) -> Dict[str, float]:
        """Calculate monthly return breakdown"""
        monthly_returns = {}
        
        for trade in trades:
            if trade.pnl is not None and trade.exit_time:
                month_key = trade.exit_time.strftime("%Y-%m")
                if month_key not in monthly_returns:
                    monthly_returns[month_key] = 0.0
                monthly_returns[month_key] += trade.pnl
        
        return monthly_returns
    
    def calculate_drawdown_analysis(self, equity_curve: List[float]) -> Dict[str, Any]:
        """Detailed drawdown analysis"""
        if len(equity_curve) < 2:
            return {"max_drawdown": 0.0, "avg_drawdown": 0.0, "drawdown_duration": 0}
        
        drawdowns = []
        running_max = equity_curve[0]
        in_drawdown = False
        drawdown_start = 0
        
        for i, equity in enumerate(equity_curve):
            if equity > running_max:
                if in_drawdown:
                    # Drawdown ended
                    drawdown_depth = (running_max - min(equity_curve[drawdown_start:i])) / running_max
                    drawdowns.append({
                        "depth": drawdown_depth,
                        "duration": i - drawdown_start,
                        "start": drawdown_start,
                        "end": i
                    })
                    in_drawdown = False
                running_max = equity
            elif equity < running_max and not in_drawdown:
                in_drawdown = True
                drawdown_start = i
        
        if not drawdowns:
            return {"max_drawdown": 0.0, "avg_drawdown": 0.0, "drawdown_duration": 0}
        
        max_drawdown = max(dd["depth"] for dd in drawdowns)
        avg_drawdown = np.mean([dd["depth"] for dd in drawdowns])
        avg_duration = np.mean([dd["duration"] for dd in drawdowns])
        
        return {
            "max_drawdown": max_drawdown,
            "avg_drawdown": avg_drawdown,
            "avg_drawdown_duration": avg_duration,
            "total_drawdown_periods": len(drawdowns)
        }