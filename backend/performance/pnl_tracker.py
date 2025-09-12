"""
Real Performance & P&L Tracking Module
Tracks actual trading performance with spreads, commissions, and real market conditions
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func

from ..models import Signal
from ..logs.logger import get_logger
from ..database import engine

logger = get_logger(__name__)

class PnLTracker:
    """Track real P&L with spreads, commissions, and slippage"""
    
    def __init__(self):
        # Typical forex spreads (in pips) - conservative estimates
        self.typical_spreads = {
            'EURUSD': 1.2, 'GBPUSD': 1.8, 'USDJPY': 1.5,
            'AUDUSD': 1.8, 'USDCAD': 2.2, 'USDCHF': 1.9,
            'NZDUSD': 2.5, 'EURGBP': 1.5, 'EURJPY': 2.0,
            'GBPJPY': 3.0, 'AUDJPY': 2.5, 'CHFJPY': 3.5,
            'EURCHF': 2.2, 'GBPAUD': 4.0, 'AUDCAD': 3.5
        }
        
        # Typical commission per lot (varies by broker)
        self.commission_per_lot = 3.5  # USD per round turn
        
        # Average slippage (in pips)
        self.average_slippage = 0.8
        
        # Lot size values
        self.standard_lot = 100000
        
    def calculate_pip_value(self, symbol: str, lot_size: float = 0.01) -> float:
        """
        Calculate pip value for a given symbol and lot size
        
        Args:
            symbol: Currency pair (e.g., 'EURUSD')
            lot_size: Position size in lots (0.01 = micro lot)
            
        Returns:
            Pip value in USD
        """
        # For pairs where USD is quote currency, 1 pip = $0.10 per micro lot
        if symbol.endswith('USD'):
            return lot_size * 10.0
        
        # For JPY pairs, pip value is different (0.01 instead of 0.0001)
        elif symbol.endswith('JPY'):
            if symbol.startswith('USD'):
                # For USDJPY, need to convert JPY to USD
                return lot_size * 10.0 / 149.50  # Approximate rate
            else:
                # For EURJPY, GBPJPY, etc.
                return lot_size * 10.0 / 149.50
        
        # For other pairs, estimate using approximate rates
        else:
            return lot_size * 10.0  # Simplified approximation
    
    def calculate_real_pnl(
        self, 
        signal: Signal, 
        exit_price: float,
        lot_size: float = 0.01
    ) -> Dict[str, Any]:
        """
        Calculate real P&L including all costs
        
        Args:
            signal: Signal object with entry details
            exit_price: Actual exit price
            lot_size: Position size in lots
            
        Returns:
            Dictionary with P&L breakdown
        """
        try:
            symbol = signal.symbol
            entry_price = signal.price
            action = signal.action
            
            # Get spread and costs
            spread = self.typical_spreads.get(symbol, 2.0)
            pip_value = self.calculate_pip_value(symbol, lot_size)
            commission = self.commission_per_lot * lot_size
            
            # Calculate pip difference
            if action == 'BUY':
                pip_movement = (exit_price - entry_price) * 10000
                # Account for entry spread (buy at ask) and exit spread (sell at bid)
                pip_movement -= spread + self.average_slippage
            else:  # SELL
                pip_movement = (entry_price - exit_price) * 10000
                # Account for entry spread (sell at bid) and exit spread (buy at ask)
                pip_movement -= spread + self.average_slippage
            
            # Calculate gross and net P&L
            gross_pnl = pip_movement * pip_value
            net_pnl = gross_pnl - commission
            
            return {
                'gross_pnl': round(gross_pnl, 2),
                'net_pnl': round(net_pnl, 2),
                'pip_movement': round(pip_movement, 1),
                'spread_cost': round(spread * pip_value, 2),
                'slippage_cost': round(self.average_slippage * pip_value, 2),
                'commission': round(commission, 2),
                'total_costs': round((spread + self.average_slippage) * pip_value + commission, 2),
                'lot_size': lot_size,
                'pip_value': round(pip_value, 2)
            }
            
        except Exception as e:
            logger.error(f"Error calculating P&L for signal {signal.id}: {e}")
            return {
                'gross_pnl': 0.0,
                'net_pnl': 0.0,
                'pip_movement': 0.0,
                'error': str(e)
            }
    
    def get_strategy_performance(self, strategy: str, days: int = 30, db: Session = None) -> Dict[str, Any]:
        """Get performance metrics for a specific strategy"""
        try:
            if not db:
                from sqlalchemy.orm import sessionmaker
                SessionLocal = sessionmaker(bind=engine)
                db = SessionLocal()
                close_session = True
            else:
                close_session = False
                
            # Get signals from last N days
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            signals = db.query(Signal).filter(
                Signal.strategy == strategy,
                Signal.issued_at >= cutoff_date,
                Signal.result.in_(['WIN', 'LOSS']),
                Signal.pips_result.isnot(None)
            ).all()
            
            if not signals:
                return {
                    'total_signals': 0,
                    'win_rate': 0.0,
                    'avg_pips': 0.0,
                    'total_pips': 0.0,
                    'best_trade': 0.0,
                    'worst_trade': 0.0,
                    'estimated_pnl': 0.0
                }
            
            # Calculate metrics
            total_signals = len(signals)
            winning_signals = [s for s in signals if s.result == 'WIN']
            win_rate = len(winning_signals) / total_signals * 100
            
            pips_results = [s.pips_result for s in signals]
            avg_pips = np.mean(pips_results)
            total_pips = sum(pips_results)
            best_trade = max(pips_results)
            worst_trade = min(pips_results)
            
            # Estimate P&L with standard micro lot
            estimated_pnl = 0.0
            for signal in signals:
                symbol = signal.symbol
                pip_value = self.calculate_pip_value(symbol, 0.01)  # Micro lot
                estimated_pnl += signal.pips_result * pip_value
            
            # Subtract estimated costs
            total_spread_cost = total_signals * np.mean(list(self.typical_spreads.values())) * 0.10
            total_commission = total_signals * self.commission_per_lot * 0.01
            estimated_pnl -= (total_spread_cost + total_commission)
            
            return {
                'total_signals': total_signals,
                'win_rate': round(win_rate, 1),
                'avg_pips': round(avg_pips, 1),
                'total_pips': round(total_pips, 1),
                'best_trade': round(best_trade, 1),
                'worst_trade': round(worst_trade, 1),
                'estimated_pnl': round(estimated_pnl, 2),
                'avg_pnl_per_trade': round(estimated_pnl / total_signals, 2) if total_signals > 0 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error calculating strategy performance for {strategy}: {e}")
            return {'error': str(e)}
        finally:
            if close_session and 'db' in locals():
                db.close()
    
    def get_portfolio_performance(self, days: int = 30, db: Session = None) -> Dict[str, Any]:
        """Get overall portfolio performance metrics"""
        try:
            if not db:
                from sqlalchemy.orm import sessionmaker
                SessionLocal = sessionmaker(bind=engine)
                db = SessionLocal()
                close_session = True
            else:
                close_session = False
                
            # Get all completed signals from last N days
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            signals = db.query(Signal).filter(
                Signal.issued_at >= cutoff_date,
                Signal.result.in_(['WIN', 'LOSS']),
                Signal.pips_result.isnot(None)
            ).all()
            
            if not signals:
                return {
                    'total_signals': 0,
                    'overall_win_rate': 0.0,
                    'total_pips': 0.0,
                    'estimated_total_pnl': 0.0,
                    'strategy_breakdown': {},
                    'symbol_breakdown': {}
                }
            
            # Overall metrics
            total_signals = len(signals)
            winning_signals = len([s for s in signals if s.result == 'WIN'])
            overall_win_rate = winning_signals / total_signals * 100
            total_pips = sum(s.pips_result for s in signals)
            
            # Strategy breakdown
            strategy_breakdown = {}
            strategies = set(s.strategy for s in signals)
            for strategy in strategies:
                strategy_breakdown[strategy] = self.get_strategy_performance(strategy, days, db)
            
            # Symbol breakdown
            symbol_breakdown = {}
            symbols = set(s.symbol for s in signals)
            for symbol in symbols:
                symbol_signals = [s for s in signals if s.symbol == symbol]
                symbol_pips = sum(s.pips_result for s in symbol_signals)
                symbol_win_rate = len([s for s in symbol_signals if s.result == 'WIN']) / len(symbol_signals) * 100
                
                symbol_breakdown[symbol] = {
                    'signals': len(symbol_signals),
                    'win_rate': round(symbol_win_rate, 1),
                    'total_pips': round(symbol_pips, 1)
                }
            
            # Estimate total P&L
            estimated_total_pnl = 0.0
            for signal in signals:
                pip_value = self.calculate_pip_value(signal.symbol, 0.01)
                estimated_total_pnl += signal.pips_result * pip_value
            
            # Subtract costs
            total_costs = total_signals * (np.mean(list(self.typical_spreads.values())) * 0.10 + self.commission_per_lot * 0.01)
            estimated_total_pnl -= total_costs
            
            return {
                'total_signals': total_signals,
                'overall_win_rate': round(overall_win_rate, 1),
                'total_pips': round(total_pips, 1),
                'estimated_total_pnl': round(estimated_total_pnl, 2),
                'avg_pnl_per_trade': round(estimated_total_pnl / total_signals, 2) if total_signals > 0 else 0.0,
                'strategy_breakdown': strategy_breakdown,
                'symbol_breakdown': symbol_breakdown
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio performance: {e}")
            return {'error': str(e)}
        finally:
            if close_session and 'db' in locals():
                db.close()

# Global P&L tracker instance
pnl_tracker = PnLTracker()