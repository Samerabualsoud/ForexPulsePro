"""
Advanced Strategy Performance Manager with Multi-Timeframe Analysis
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import structlog
from sqlalchemy.orm import Session
from collections import defaultdict, deque

from backend.database import get_session
from backend.models import Signal, Strategy
from backend.analysis.backtester import AdvancedBacktester, Trade, StrategyMetrics

logger = structlog.get_logger(__name__)

class StrategyPerformanceTracker:
    """Track and optimize strategy performance in real-time"""
    
    def __init__(self):
        self.performance_cache: Dict[str, Dict] = {}
        self.confidence_adjustments: Dict[str, float] = {}
        self.recent_performance: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        
    def update_strategy_performance(self, signal_data: Dict[str, Any]) -> None:
        """Update strategy performance based on signal results"""
        strategy_name = signal_data.get('strategy')
        if not strategy_name:
            return
            
        try:
            # Add to recent performance tracking
            performance_entry = {
                'timestamp': datetime.now(),
                'symbol': signal_data.get('symbol'),
                'confidence': signal_data.get('confidence', 0.0),
                'result': signal_data.get('result'),  # 'profit', 'loss', 'pending'
                'pnl': signal_data.get('pnl', 0.0)
            }
            
            self.recent_performance[strategy_name].append(performance_entry)
            
            # Calculate rolling metrics
            self._calculate_rolling_metrics(strategy_name)
            
            logger.debug(f"Updated performance for strategy {strategy_name}")
            
        except Exception as e:
            logger.error(f"Error updating strategy performance: {e}")
    
    def _calculate_rolling_metrics(self, strategy_name: str) -> None:
        """Calculate rolling performance metrics for dynamic adjustments"""
        recent = list(self.recent_performance[strategy_name])
        if len(recent) < 5:  # Need minimum sample size
            return
        
        # Calculate recent win rate
        closed_trades = [r for r in recent if r['result'] in ['profit', 'loss']]
        if not closed_trades:
            return
            
        wins = len([r for r in closed_trades if r['result'] == 'profit'])
        win_rate = wins / len(closed_trades)
        
        # Calculate average confidence vs results
        confidence_results = [(r['confidence'], 1 if r['result'] == 'profit' else 0) for r in closed_trades]
        
        # Dynamic confidence adjustment based on recent performance
        if win_rate > 0.7:  # Strong performance
            adjustment = min(0.1, (win_rate - 0.7) * 0.5)
        elif win_rate < 0.4:  # Poor performance
            adjustment = max(-0.2, (win_rate - 0.4) * 0.5)
        else:
            adjustment = 0.0
            
        self.confidence_adjustments[strategy_name] = adjustment
        
        # Cache performance metrics
        avg_pnl = np.mean([r['pnl'] for r in closed_trades if r['pnl']])
        
        self.performance_cache[strategy_name] = {
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'confidence_adjustment': adjustment,
            'sample_size': len(closed_trades),
            'last_updated': datetime.now()
        }
    
    def get_strategy_multiplier(self, strategy_name: str) -> float:
        """Get performance-based confidence multiplier"""
        if strategy_name not in self.performance_cache:
            return 1.0  # Neutral
        
        performance = self.performance_cache[strategy_name]
        base_multiplier = 1.0
        
        # Adjust based on recent win rate
        win_rate = performance.get('win_rate', 0.5)
        if win_rate > 0.7:
            base_multiplier += 0.15  # Boost high performers
        elif win_rate < 0.4:
            base_multiplier -= 0.25  # Reduce poor performers
        
        # Adjust based on sample size (more confident with more data)
        sample_size = performance.get('sample_size', 0)
        if sample_size >= 20:
            confidence_factor = 1.0
        elif sample_size >= 10:
            confidence_factor = 0.8
        else:
            confidence_factor = 0.6
            
        return base_multiplier * confidence_factor
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        summary = {
            'strategy_performance': {},
            'top_performers': [],
            'underperformers': [],
            'total_strategies': len(self.performance_cache)
        }
        
        for strategy, metrics in self.performance_cache.items():
            summary['strategy_performance'][strategy] = {
                'win_rate': round(metrics.get('win_rate', 0) * 100, 1),
                'avg_pnl': round(metrics.get('avg_pnl', 0), 4),
                'confidence_multiplier': round(self.get_strategy_multiplier(strategy), 2),
                'trades_analyzed': metrics.get('sample_size', 0),
                'status': 'active' if metrics.get('win_rate', 0) > 0.3 else 'underperforming'
            }
        
        # Identify top and bottom performers
        if self.performance_cache:
            sorted_strategies = sorted(
                self.performance_cache.items(),
                key=lambda x: x[1].get('win_rate', 0),
                reverse=True
            )
            
            summary['top_performers'] = [s[0] for s in sorted_strategies[:3]]
            summary['underperformers'] = [s[0] for s in sorted_strategies[-2:] if s[1].get('win_rate', 1) < 0.4]
        
        return summary

class MultiTimeframeAnalyzer:
    """Multi-timeframe confirmation system for enhanced signal quality"""
    
    def __init__(self):
        self.timeframes = ['1H', '4H', '1D']
        self.confirmation_weights = {'1H': 0.3, '4H': 0.4, '1D': 0.3}
    
    def analyze_multiple_timeframes(self, symbol: str, base_signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze signal across multiple timeframes for confirmation
        
        Args:
            symbol: Currency pair
            base_signal: Original signal from primary timeframe
            
        Returns:
            Multi-timeframe analysis with confirmation score
        """
        try:
            # Generate market data for different timeframes
            timeframe_analysis = {}
            
            for tf in self.timeframes:
                tf_data = self._generate_timeframe_data(symbol, tf)
                tf_signal = self._analyze_timeframe(tf_data, base_signal['strategy'])
                
                timeframe_analysis[tf] = {
                    'signal': tf_signal.get('action', 'NEUTRAL'),
                    'confidence': tf_signal.get('confidence', 0.5),
                    'trend': tf_signal.get('trend', 'SIDEWAYS'),
                    'strength': tf_signal.get('strength', 0.5)
                }
            
            # Calculate confirmation score
            confirmation_score = self._calculate_confirmation_score(base_signal, timeframe_analysis)
            
            # Generate recommendation
            recommendation = self._generate_recommendation(confirmation_score, timeframe_analysis)
            
            result = {
                'symbol': symbol,
                'base_signal': base_signal,
                'timeframe_analysis': timeframe_analysis,
                'confirmation_score': confirmation_score,
                'recommendation': recommendation,
                'analyzed_at': datetime.now().isoformat()
            }
            
            logger.debug(f"Multi-timeframe analysis for {symbol}: confirmation={confirmation_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in multi-timeframe analysis: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'recommendation': 'SKIP'
            }
    
    def _generate_timeframe_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Generate synthetic market data for different timeframes"""
        # In production, this would call actual data providers
        periods = {'1H': 168, '4H': 42, '1D': 14}  # 1 week of data
        
        base_price = {"EURUSD": 1.0894, "GBPUSD": 1.3156, "USDJPY": 149.85}.get(symbol, 1.0)
        
        # Generate realistic OHLC data
        np.random.seed(hash(symbol + timeframe) % 1000)
        
        dates = pd.date_range(end=datetime.now(), periods=periods[timeframe], freq=timeframe)
        returns = np.random.normal(0, 0.002, periods[timeframe])  # Higher volatility for longer TF
        prices = base_price * np.cumprod(1 + returns)
        
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.001)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.001)))
            
            data.append({
                'timestamp': date,
                'open': round(open_price, 5),
                'high': round(high_price, 5),
                'low': round(low_price, 5),
                'close': round(close_price, 5),
                'volume': np.random.randint(1000, 10000)
            })
        
        df = pd.DataFrame(data)
        return df.set_index('timestamp')
    
    def _analyze_timeframe(self, data: pd.DataFrame, strategy: str) -> Dict[str, Any]:
        """Analyze single timeframe for signals"""
        try:
            # Simple trend analysis
            closes = data['close'].values
            if len(closes) < 10:
                return {'action': 'NEUTRAL', 'confidence': 0.5}
            
            # Calculate moving averages
            ma_short = np.mean(closes[-5:])
            ma_long = np.mean(closes[-10:])
            
            # Determine trend
            if ma_short > ma_long * 1.001:  # 0.1% buffer
                trend = 'BULLISH'
                action = 'BUY'
            elif ma_short < ma_long * 0.999:
                trend = 'BEARISH'
                action = 'SELL'
            else:
                trend = 'SIDEWAYS'
                action = 'NEUTRAL'
            
            # Calculate confidence based on trend strength
            trend_strength = abs(ma_short - ma_long) / ma_long
            confidence = min(0.9, 0.5 + trend_strength * 100)
            
            return {
                'action': action,
                'confidence': confidence,
                'trend': trend,
                'strength': trend_strength,
                'ma_short': ma_short,
                'ma_long': ma_long
            }
            
        except Exception as e:
            logger.error(f"Error analyzing timeframe: {e}")
            return {'action': 'NEUTRAL', 'confidence': 0.5}
    
    def _calculate_confirmation_score(self, base_signal: Dict, timeframe_analysis: Dict) -> float:
        """Calculate multi-timeframe confirmation score"""
        base_action = base_signal.get('action', 'NEUTRAL')
        if base_action == 'NEUTRAL':
            return 0.5
        
        confirmation_score = 0.0
        total_weight = 0.0
        
        for tf, weight in self.confirmation_weights.items():
            if tf in timeframe_analysis:
                tf_data = timeframe_analysis[tf]
                tf_action = tf_data.get('signal', 'NEUTRAL')
                tf_confidence = tf_data.get('confidence', 0.5)
                
                # Check if timeframe aligns with base signal
                if tf_action == base_action:
                    alignment_score = tf_confidence
                elif tf_action == 'NEUTRAL':
                    alignment_score = 0.5
                else:
                    alignment_score = 1.0 - tf_confidence  # Opposing signal
                
                confirmation_score += alignment_score * weight
                total_weight += weight
        
        return confirmation_score / total_weight if total_weight > 0 else 0.5
    
    def _generate_recommendation(self, confirmation_score: float, timeframe_analysis: Dict) -> str:
        """Generate final recommendation based on analysis"""
        if confirmation_score >= 0.75:
            return 'STRONG_SIGNAL'
        elif confirmation_score >= 0.6:
            return 'MODERATE_SIGNAL'
        elif confirmation_score <= 0.35:
            return 'SKIP'
        else:
            return 'WEAK_SIGNAL'

# Global instances
strategy_tracker = StrategyPerformanceTracker()
mtf_analyzer = MultiTimeframeAnalyzer()