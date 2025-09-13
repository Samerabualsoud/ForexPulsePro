"""
Advanced Backtesting Framework
Comprehensive backtesting system with Monte Carlo simulation, walk-forward analysis,
and AI-enhanced validation using both Manus AI and ChatGPT
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import asyncio
import warnings
warnings.filterwarnings('ignore')

from ..logs.logger import get_logger
from ..ai_capabilities import get_ai_capabilities, OPENAI_ENABLED
from .manus_ai import ManusAI
from ..signals.utils import calculate_atr

logger = get_logger(__name__)

# Check AI capabilities and conditionally import ChatGPT components
ai_capabilities = get_ai_capabilities()
CHATGPT_AVAILABLE = ai_capabilities['openai_enabled']

# Conditional imports
ChatGPTStrategyOptimizer = None
AIStrategyConsensus = None

if CHATGPT_AVAILABLE:
    try:
        from .chatgpt_strategy_optimizer import ChatGPTStrategyOptimizer
        from .ai_strategy_consensus import AIStrategyConsensus
        logger.info("Advanced backtester: ChatGPT components loaded for dual-AI mode")
    except ImportError as e:
        logger.warning(f"Advanced backtester: ChatGPT components failed to load: {e}")
        CHATGPT_AVAILABLE = False
else:
    logger.info("Advanced backtester: Operating in Manus AI only mode")

class BacktestType(Enum):
    """Types of backtesting"""
    SIMPLE = "simple"
    MONTE_CARLO = "monte_carlo"
    WALK_FORWARD = "walk_forward"
    OUT_OF_SAMPLE = "out_of_sample"
    AI_ENHANCED = "ai_enhanced"

@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    position_size: float
    commission: float
    slippage: float
    monte_carlo_runs: int
    walk_forward_periods: int
    out_of_sample_ratio: float
    confidence_levels: List[float]

@dataclass
class BacktestResults:
    """Comprehensive backtest results"""
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_factor: float
    trades_count: int
    avg_trade_return: float
    volatility: float
    calmar_ratio: float
    value_at_risk: float
    conditional_var: float
    
    # AI-specific metrics
    ai_confidence_correlation: float
    strategy_consistency_score: float
    regime_adaptability: float
    
    # Monte Carlo specific
    monte_carlo_confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None
    worst_case_scenario: Optional[float] = None
    best_case_scenario: Optional[float] = None
    
    # Walk-forward specific
    walk_forward_periods: Optional[List[Dict]] = None
    out_of_sample_performance: Optional[Dict] = None

class AdvancedBacktester:
    """
    Advanced Backtesting Framework with AI integration
    Provides comprehensive backtesting capabilities including Monte Carlo simulation,
    walk-forward analysis, and AI-enhanced strategy validation
    """
    
    def __init__(self):
        self.name = "Advanced Backtesting Framework"
        
        # Initialize AI services (Manus AI always available)
        self.manus_ai = ManusAI()
        
        # Conditionally initialize ChatGPT components
        if CHATGPT_AVAILABLE and ChatGPTStrategyOptimizer and AIStrategyConsensus:
            try:
                self.chatgpt_optimizer = ChatGPTStrategyOptimizer()
                self.ai_consensus = AIStrategyConsensus()
                logger.info("AdvancedBacktester: Dual-AI mode active (Manus AI + ChatGPT)")
            except Exception as e:
                logger.warning(f"AdvancedBacktester: Failed to initialize ChatGPT components: {e}")
                self.chatgpt_optimizer = None
                self.ai_consensus = None
        else:
            self.chatgpt_optimizer = None
            self.ai_consensus = None
            logger.info("AdvancedBacktester: Single-AI mode active (Manus AI only)")
        
        # Default configuration
        self.default_config = BacktestConfig(
            start_date=datetime.now() - timedelta(days=365),
            end_date=datetime.now(),
            initial_capital=100000.0,
            position_size=0.01,  # 1% risk per trade
            commission=0.0002,   # 0.02% commission
            slippage=0.0001,     # 0.01% slippage
            monte_carlo_runs=1000,
            walk_forward_periods=12,
            out_of_sample_ratio=0.3,
            confidence_levels=[0.95, 0.99]
        )
        
        # Risk-free rate for Sharpe calculation (annual)
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        
        # Strategy performance cache for efficiency
        self.strategy_cache = {}
        
        logger.info("Advanced Backtesting Framework initialized")
    
    async def run_comprehensive_backtest(
        self,
        symbol: str,
        strategy_name: str,
        strategy_config: Dict[str, Any],
        market_data: pd.DataFrame,
        config: Optional[BacktestConfig] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive backtesting including all advanced methods
        
        Args:
            symbol: Trading symbol
            strategy_name: Name of strategy to test
            strategy_config: Strategy configuration parameters
            market_data: Historical price data
            config: Backtesting configuration
            
        Returns:
            Dict with comprehensive backtest results
        """
        try:
            logger.info(f"Starting comprehensive backtest for {symbol} using {strategy_name}")
            
            if config is None:
                config = self.default_config
            
            # Validate data and configuration
            validated_data = self._validate_backtest_data(market_data, config)
            if validated_data is None:
                raise ValueError("Invalid backtest data or configuration")
            
            # Run all backtest types concurrently for efficiency
            backtest_tasks = []
            
            # 1. Simple backtest
            simple_task = self._run_simple_backtest(
                symbol, strategy_name, strategy_config, validated_data, config
            )
            backtest_tasks.append(("simple", simple_task))
            
            # 2. Monte Carlo simulation
            monte_carlo_task = self._run_monte_carlo_backtest(
                symbol, strategy_name, strategy_config, validated_data, config
            )
            backtest_tasks.append(("monte_carlo", monte_carlo_task))
            
            # 3. Walk-forward analysis
            walk_forward_task = self._run_walk_forward_backtest(
                symbol, strategy_name, strategy_config, validated_data, config
            )
            backtest_tasks.append(("walk_forward", walk_forward_task))
            
            # 4. AI-enhanced backtesting
            ai_enhanced_task = self._run_ai_enhanced_backtest(
                symbol, strategy_name, strategy_config, validated_data, config
            )
            backtest_tasks.append(("ai_enhanced", ai_enhanced_task))
            
            # Execute all backtests
            results = {}
            for backtest_type, task in backtest_tasks:
                try:
                    result = await task
                    results[backtest_type] = result
                    logger.info(f"Completed {backtest_type} backtest for {symbol}")
                except Exception as e:
                    logger.error(f"Error in {backtest_type} backtest: {e}")
                    results[backtest_type] = {'error': str(e)}
            
            # Generate comprehensive analysis
            comprehensive_analysis = self._generate_comprehensive_analysis(results, symbol, strategy_name)
            
            # Calculate overall strategy score
            overall_score = self._calculate_overall_strategy_score(results)
            
            final_results = {
                'symbol': symbol,
                'strategy_name': strategy_name,
                'timestamp': datetime.utcnow().isoformat(),
                'backtest_results': results,
                'comprehensive_analysis': comprehensive_analysis,
                'overall_strategy_score': overall_score,
                'recommendation': self._generate_strategy_recommendation(overall_score, results),
                'risk_assessment': self._generate_risk_assessment(results),
                'performance_summary': self._generate_performance_summary(results)
            }
            
            logger.info(f"Comprehensive backtest completed for {symbol} {strategy_name} "
                       f"(score: {overall_score:.2f})")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in comprehensive backtest for {symbol} {strategy_name}: {e}")
            return {
                'error': str(e),
                'symbol': symbol,
                'strategy_name': strategy_name,
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def optimize_strategy_parameters(
        self,
        symbol: str,
        strategy_name: str,
        parameter_ranges: Dict[str, Tuple[float, float, float]],  # (min, max, step)
        market_data: pd.DataFrame,
        optimization_metric: str = 'sharpe_ratio'
    ) -> Dict[str, Any]:
        """
        AI-enhanced parameter optimization using advanced backtesting
        
        Args:
            symbol: Trading symbol
            strategy_name: Strategy to optimize
            parameter_ranges: Dict of parameter ranges for optimization
            market_data: Historical data
            optimization_metric: Metric to optimize (sharpe_ratio, total_return, etc.)
            
        Returns:
            Dict with optimized parameters and performance
        """
        try:
            logger.info(f"Starting parameter optimization for {symbol} {strategy_name}")
            
            # Generate parameter combinations for testing
            parameter_combinations = self._generate_parameter_combinations(parameter_ranges)
            
            # Limit combinations to prevent excessive computation
            max_combinations = min(len(parameter_combinations), 100)
            selected_combinations = parameter_combinations[:max_combinations]
            
            logger.info(f"Testing {len(selected_combinations)} parameter combinations")
            
            # Test each parameter combination
            optimization_results = []
            
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_params = {}
                
                for params in selected_combinations:
                    future = executor.submit(
                        self._test_parameter_combination,
                        symbol, strategy_name, params, market_data
                    )
                    future_to_params[future] = params
                
                for future in concurrent.futures.as_completed(future_to_params):
                    params = future_to_params[future]
                    try:
                        result = future.result()
                        if result and 'error' not in result:
                            result['parameters'] = params
                            optimization_results.append(result)
                    except Exception as e:
                        logger.warning(f"Error testing parameters {params}: {e}")
            
            if not optimization_results:
                raise ValueError("No valid optimization results found")
            
            # Find best parameters based on optimization metric
            best_result = max(
                optimization_results,
                key=lambda x: x.get(optimization_metric, 0)
            )
            
            # Get AI recommendations for the best parameters
            ai_validation = await self._validate_optimized_parameters(
                symbol, strategy_name, best_result['parameters'], market_data
            )
            
            # Generate optimization report
            optimization_report = self._generate_optimization_report(
                optimization_results, best_result, optimization_metric
            )
            
            result = {
                'symbol': symbol,
                'strategy_name': strategy_name,
                'timestamp': datetime.utcnow().isoformat(),
                'optimization_metric': optimization_metric,
                'best_parameters': best_result['parameters'],
                'best_performance': {
                    optimization_metric: best_result.get(optimization_metric, 0),
                    'total_return': best_result.get('total_return', 0),
                    'sharpe_ratio': best_result.get('sharpe_ratio', 0),
                    'max_drawdown': best_result.get('max_drawdown', 0)
                },
                'ai_validation': ai_validation,
                'optimization_report': optimization_report,
                'tested_combinations': len(optimization_results),
                'performance_improvement': self._calculate_performance_improvement(optimization_results)
            }
            
            logger.info(f"Parameter optimization completed for {symbol} {strategy_name}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in parameter optimization for {symbol} {strategy_name}: {e}")
            return {
                'error': str(e),
                'symbol': symbol,
                'strategy_name': strategy_name
            }
    
    async def compare_strategies(
        self,
        symbol: str,
        strategies: List[Dict[str, Any]],  # List of {name, config} dicts
        market_data: pd.DataFrame,
        comparison_metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple strategies using advanced backtesting
        
        Args:
            symbol: Trading symbol
            strategies: List of strategies to compare
            market_data: Historical price data
            comparison_metrics: Metrics to use for comparison
            
        Returns:
            Dict with strategy comparison results
        """
        try:
            logger.info(f"Comparing {len(strategies)} strategies for {symbol}")
            
            if comparison_metrics is None:
                comparison_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
            
            # Run backtests for all strategies
            strategy_results = {}
            
            for strategy in strategies:
                strategy_name = strategy['name']
                strategy_config = strategy['config']
                
                try:
                    result = await self.run_comprehensive_backtest(
                        symbol, strategy_name, strategy_config, market_data
                    )
                    strategy_results[strategy_name] = result
                except Exception as e:
                    logger.warning(f"Error backtesting strategy {strategy_name}: {e}")
                    strategy_results[strategy_name] = {'error': str(e)}
            
            # Generate comparison analysis
            comparison_analysis = self._generate_strategy_comparison(
                strategy_results, comparison_metrics
            )
            
            # Get AI consensus on best strategy
            ai_recommendation = await self._get_ai_strategy_recommendation(
                symbol, strategy_results, market_data
            )
            
            result = {
                'symbol': symbol,
                'timestamp': datetime.utcnow().isoformat(),
                'strategies_tested': len(strategies),
                'strategy_results': strategy_results,
                'comparison_analysis': comparison_analysis,
                'ai_recommendation': ai_recommendation,
                'best_strategy': comparison_analysis.get('best_overall_strategy'),
                'performance_rankings': comparison_analysis.get('performance_rankings', {}),
                'risk_rankings': comparison_analysis.get('risk_rankings', {})
            }
            
            logger.info(f"Strategy comparison completed for {symbol}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error comparing strategies for {symbol}: {e}")
            return {
                'error': str(e),
                'symbol': symbol,
                'strategies_tested': len(strategies)
            }
    
    async def _run_simple_backtest(
        self,
        symbol: str,
        strategy_name: str,
        strategy_config: Dict,
        market_data: pd.DataFrame,
        config: BacktestConfig
    ) -> BacktestResults:
        """Run simple backtest with basic metrics"""
        try:
            # Simulate strategy signals and trades
            trades = self._simulate_strategy_trades(strategy_name, strategy_config, market_data)
            
            if not trades:
                return self._create_empty_results("No trades generated")
            
            # Calculate performance metrics
            returns = self._calculate_returns(trades, config)
            metrics = self._calculate_performance_metrics(returns, trades, config)
            
            return BacktestResults(**metrics)
            
        except Exception as e:
            logger.warning(f"Error in simple backtest: {e}")
            return self._create_empty_results(str(e))
    
    async def _run_monte_carlo_backtest(
        self,
        symbol: str,
        strategy_name: str,
        strategy_config: Dict,
        market_data: pd.DataFrame,
        config: BacktestConfig
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation backtest"""
        try:
            logger.info(f"Running Monte Carlo simulation with {config.monte_carlo_runs} runs")
            
            # Generate multiple scenarios by bootstrapping returns
            base_returns = market_data['close'].pct_change().dropna()
            
            monte_carlo_results = []
            
            for run in range(config.monte_carlo_runs):
                # Bootstrap returns to create new scenario
                scenario_returns = np.random.choice(base_returns, size=len(base_returns), replace=True)
                
                # Create scenario price data
                scenario_prices = self._create_scenario_data(market_data, scenario_returns)
                
                # Run backtest on scenario
                trades = self._simulate_strategy_trades(strategy_name, strategy_config, scenario_prices)
                
                if trades:
                    returns = self._calculate_returns(trades, config)
                    total_return = (np.prod(1 + returns) - 1) * 100
                    monte_carlo_results.append(total_return)
            
            if not monte_carlo_results:
                return {'error': 'No valid Monte Carlo results'}
            
            # Calculate Monte Carlo statistics
            mc_results = np.array(monte_carlo_results)
            
            result = {
                'mean_return': float(np.mean(mc_results)),
                'median_return': float(np.median(mc_results)),
                'std_return': float(np.std(mc_results)),
                'worst_case': float(np.min(mc_results)),
                'best_case': float(np.max(mc_results)),
                'var_95': float(np.percentile(mc_results, 5)),
                'var_99': float(np.percentile(mc_results, 1)),
                'confidence_intervals': {
                    '95%': (float(np.percentile(mc_results, 2.5)), float(np.percentile(mc_results, 97.5))),
                    '99%': (float(np.percentile(mc_results, 0.5)), float(np.percentile(mc_results, 99.5)))
                },
                'probability_positive': float(np.mean(mc_results > 0)),
                'runs_completed': len(monte_carlo_results)
            }
            
            return result
            
        except Exception as e:
            logger.warning(f"Error in Monte Carlo backtest: {e}")
            return {'error': str(e)}
    
    async def _run_walk_forward_backtest(
        self,
        symbol: str,
        strategy_name: str,
        strategy_config: Dict,
        market_data: pd.DataFrame,
        config: BacktestConfig
    ) -> Dict[str, Any]:
        """Run walk-forward analysis"""
        try:
            total_periods = len(market_data)
            period_size = total_periods // config.walk_forward_periods
            
            if period_size < 30:  # Need minimum data for each period
                return {'error': 'Insufficient data for walk-forward analysis'}
            
            walk_forward_results = []
            
            for period in range(config.walk_forward_periods):
                start_idx = period * period_size
                end_idx = min(start_idx + period_size, total_periods)
                
                period_data = market_data.iloc[start_idx:end_idx].copy()
                
                # Run backtest on period
                trades = self._simulate_strategy_trades(strategy_name, strategy_config, period_data)
                
                if trades:
                    returns = self._calculate_returns(trades, config)
                    period_metrics = self._calculate_performance_metrics(returns, trades, config)
                    
                    walk_forward_results.append({
                        'period': period + 1,
                        'start_date': period_data.index[0].isoformat() if len(period_data) > 0 else None,
                        'end_date': period_data.index[-1].isoformat() if len(period_data) > 0 else None,
                        'total_return': period_metrics.get('total_return', 0),
                        'sharpe_ratio': period_metrics.get('sharpe_ratio', 0),
                        'max_drawdown': period_metrics.get('max_drawdown', 0),
                        'trades_count': len(trades)
                    })
            
            if not walk_forward_results:
                return {'error': 'No valid walk-forward results'}
            
            # Calculate walk-forward statistics
            returns_list = [r['total_return'] for r in walk_forward_results]
            sharpe_list = [r['sharpe_ratio'] for r in walk_forward_results if r['sharpe_ratio'] != 0]
            
            result = {
                'periods_tested': len(walk_forward_results),
                'period_results': walk_forward_results,
                'average_return': float(np.mean(returns_list)) if returns_list else 0,
                'return_stability': float(np.std(returns_list)) if len(returns_list) > 1 else 0,
                'average_sharpe': float(np.mean(sharpe_list)) if sharpe_list else 0,
                'consistent_performance': float(np.mean([r > 0 for r in returns_list])) if returns_list else 0,
                'best_period_return': float(max(returns_list)) if returns_list else 0,
                'worst_period_return': float(min(returns_list)) if returns_list else 0
            }
            
            return result
            
        except Exception as e:
            logger.warning(f"Error in walk-forward backtest: {e}")
            return {'error': str(e)}
    
    async def _run_ai_enhanced_backtest(
        self,
        symbol: str,
        strategy_name: str,
        strategy_config: Dict,
        market_data: pd.DataFrame,
        config: BacktestConfig
    ) -> Dict[str, Any]:
        """Run AI-enhanced backtesting with consensus validation"""
        try:
            # Get AI consensus on strategy suitability
            available_strategies = [strategy_name]
            consensus_result = await self.ai_consensus.generate_consensus_recommendation(
                symbol, market_data, available_strategies
            )
            
            # Get ChatGPT analysis of strategy
            chatgpt_analysis = await self.chatgpt_optimizer.analyze_market_conditions(
                symbol, market_data, available_strategies
            )
            
            # Get Manus AI analysis
            manus_analysis = self.manus_ai.suggest_strategies(symbol, market_data)
            
            # Run backtest with AI-enhanced signals
            enhanced_trades = self._simulate_ai_enhanced_trades(
                strategy_name, strategy_config, market_data, consensus_result
            )
            
            if not enhanced_trades:
                return {
                    'error': 'No trades generated with AI enhancement',
                    'ai_consensus': consensus_result.__dict__ if hasattr(consensus_result, '__dict__') else {}
                }
            
            # Calculate enhanced metrics
            returns = self._calculate_returns(enhanced_trades, config)
            base_metrics = self._calculate_performance_metrics(returns, enhanced_trades, config)
            
            # Add AI-specific metrics
            ai_metrics = self._calculate_ai_enhanced_metrics(
                enhanced_trades, consensus_result, chatgpt_analysis, manus_analysis
            )
            
            result = {
                **base_metrics,
                'ai_enhanced_metrics': ai_metrics,
                'ai_consensus_score': consensus_result.overall_confidence if hasattr(consensus_result, 'overall_confidence') else 0.5,
                'consensus_level': consensus_result.consensus_level.value if hasattr(consensus_result, 'consensus_level') else 'unknown',
                'ai_agreement_score': consensus_result.agreement_score if hasattr(consensus_result, 'agreement_score') else 0.5,
                'strategy_ai_suitability': ai_metrics.get('strategy_suitability_score', 0.5)
            }
            
            return result
            
        except Exception as e:
            logger.warning(f"Error in AI-enhanced backtest: {e}")
            return {'error': str(e)}
    
    def _validate_backtest_data(self, market_data: pd.DataFrame, config: BacktestConfig) -> Optional[pd.DataFrame]:
        """Validate and prepare data for backtesting"""
        try:
            if market_data is None or len(market_data) < 30:
                return None
            
            # Ensure required columns
            required_columns = ['open', 'high', 'low', 'close']
            if not all(col in market_data.columns for col in required_columns):
                return None
            
            # Remove any NaN values
            clean_data = market_data[required_columns].dropna()
            
            # Ensure data is sorted by index (time)
            clean_data = clean_data.sort_index()
            
            # Filter by date range if specified
            if hasattr(config, 'start_date') and hasattr(config, 'end_date'):
                if config.start_date and config.end_date:
                    mask = (clean_data.index >= config.start_date) & (clean_data.index <= config.end_date)
                    clean_data = clean_data[mask]
            
            return clean_data if len(clean_data) >= 30 else None
            
        except Exception as e:
            logger.warning(f"Error validating backtest data: {e}")
            return None
    
    def _simulate_strategy_trades(
        self,
        strategy_name: str,
        strategy_config: Dict,
        market_data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Simulate trading signals and generate trades"""
        try:
            trades = []
            
            # Simple trading simulation based on strategy type
            # This is a simplified implementation - in practice, you'd use the actual strategy classes
            
            if strategy_name == 'ema_rsi':
                trades = self._simulate_ema_rsi_trades(market_data, strategy_config)
            elif strategy_name == 'meanrev_bb':
                trades = self._simulate_bb_trades(market_data, strategy_config)
            elif strategy_name == 'donchian_atr':
                trades = self._simulate_donchian_trades(market_data, strategy_config)
            else:
                # Generic momentum strategy simulation
                trades = self._simulate_generic_trades(market_data, strategy_config)
            
            return trades
            
        except Exception as e:
            logger.warning(f"Error simulating strategy trades: {e}")
            return []
    
    def _simulate_ema_rsi_trades(self, market_data: pd.DataFrame, config: Dict) -> List[Dict]:
        """Simulate EMA+RSI strategy trades"""
        try:
            close = market_data['close']
            
            # Calculate EMA
            ema_period = config.get('ema_period', 20)
            ema = close.ewm(span=ema_period).mean()
            
            # Calculate RSI
            rsi_period = config.get('rsi_period', 14)
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            trades = []
            position = 0
            
            for i in range(1, len(market_data)):
                if pd.isna(ema.iloc[i]) or pd.isna(rsi.iloc[i]):
                    continue
                
                # Entry conditions
                if position == 0:
                    # Buy signal: price above EMA and RSI oversold
                    if close.iloc[i] > ema.iloc[i] and rsi.iloc[i] < 30:
                        trades.append({
                            'type': 'BUY',
                            'price': close.iloc[i],
                            'timestamp': market_data.index[i],
                            'rsi': rsi.iloc[i]
                        })
                        position = 1
                    # Sell signal: price below EMA and RSI overbought
                    elif close.iloc[i] < ema.iloc[i] and rsi.iloc[i] > 70:
                        trades.append({
                            'type': 'SELL',
                            'price': close.iloc[i],
                            'timestamp': market_data.index[i],
                            'rsi': rsi.iloc[i]
                        })
                        position = -1
                
                # Exit conditions
                elif position == 1:  # Long position
                    if close.iloc[i] < ema.iloc[i] or rsi.iloc[i] > 70:
                        trades.append({
                            'type': 'SELL',
                            'price': close.iloc[i],
                            'timestamp': market_data.index[i],
                            'rsi': rsi.iloc[i]
                        })
                        position = 0
                
                elif position == -1:  # Short position
                    if close.iloc[i] > ema.iloc[i] or rsi.iloc[i] < 30:
                        trades.append({
                            'type': 'BUY',
                            'price': close.iloc[i],
                            'timestamp': market_data.index[i],
                            'rsi': rsi.iloc[i]
                        })
                        position = 0
            
            return trades
            
        except Exception as e:
            logger.warning(f"Error simulating EMA-RSI trades: {e}")
            return []
    
    def _simulate_bb_trades(self, market_data: pd.DataFrame, config: Dict) -> List[Dict]:
        """Simulate Bollinger Bands mean reversion trades"""
        try:
            close = market_data['close']
            
            # Calculate Bollinger Bands
            period = config.get('bb_period', 20)
            std_dev = config.get('bb_std', 2)
            
            sma = close.rolling(window=period).mean()
            std = close.rolling(window=period).std()
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            trades = []
            position = 0
            
            for i in range(period, len(market_data)):
                if pd.isna(upper_band.iloc[i]) or pd.isna(lower_band.iloc[i]):
                    continue
                
                # Mean reversion strategy
                if position == 0:
                    # Buy at lower band (oversold)
                    if close.iloc[i] <= lower_band.iloc[i]:
                        trades.append({
                            'type': 'BUY',
                            'price': close.iloc[i],
                            'timestamp': market_data.index[i],
                            'bb_position': (close.iloc[i] - lower_band.iloc[i]) / (upper_band.iloc[i] - lower_band.iloc[i])
                        })
                        position = 1
                    # Sell at upper band (overbought)
                    elif close.iloc[i] >= upper_band.iloc[i]:
                        trades.append({
                            'type': 'SELL',
                            'price': close.iloc[i],
                            'timestamp': market_data.index[i],
                            'bb_position': (close.iloc[i] - lower_band.iloc[i]) / (upper_band.iloc[i] - lower_band.iloc[i])
                        })
                        position = -1
                
                # Exit at middle (SMA)
                elif position != 0:
                    if abs(close.iloc[i] - sma.iloc[i]) < (upper_band.iloc[i] - sma.iloc[i]) * 0.1:
                        trades.append({
                            'type': 'BUY' if position == -1 else 'SELL',
                            'price': close.iloc[i],
                            'timestamp': market_data.index[i],
                            'bb_position': (close.iloc[i] - lower_band.iloc[i]) / (upper_band.iloc[i] - lower_band.iloc[i])
                        })
                        position = 0
            
            return trades
            
        except Exception as e:
            logger.warning(f"Error simulating BB trades: {e}")
            return []
    
    def _simulate_donchian_trades(self, market_data: pd.DataFrame, config: Dict) -> List[Dict]:
        """Simulate Donchian channel breakout trades"""
        try:
            high = market_data['high']
            low = market_data['low']
            close = market_data['close']
            
            # Calculate Donchian channels
            period = config.get('donchian_period', 20)
            upper_channel = high.rolling(window=period).max()
            lower_channel = low.rolling(window=period).min()
            
            # Calculate ATR for position sizing
            atr = calculate_atr(high, low, close, period=14)
            
            trades = []
            position = 0
            
            for i in range(period, len(market_data)):
                if pd.isna(upper_channel.iloc[i]) or pd.isna(lower_channel.iloc[i]):
                    continue
                
                # Breakout strategy
                if position == 0:
                    # Buy on upper channel breakout
                    if high.iloc[i] > upper_channel.iloc[i-1]:
                        trades.append({
                            'type': 'BUY',
                            'price': close.iloc[i],
                            'timestamp': market_data.index[i],
                            'atr': atr.iloc[i] if not pd.isna(atr.iloc[i]) else 0.01
                        })
                        position = 1
                    # Sell on lower channel breakout
                    elif low.iloc[i] < lower_channel.iloc[i-1]:
                        trades.append({
                            'type': 'SELL',
                            'price': close.iloc[i],
                            'timestamp': market_data.index[i],
                            'atr': atr.iloc[i] if not pd.isna(atr.iloc[i]) else 0.01
                        })
                        position = -1
                
                # Exit on opposite channel touch
                elif position == 1:  # Long position
                    if low.iloc[i] <= lower_channel.iloc[i]:
                        trades.append({
                            'type': 'SELL',
                            'price': close.iloc[i],
                            'timestamp': market_data.index[i],
                            'atr': atr.iloc[i] if not pd.isna(atr.iloc[i]) else 0.01
                        })
                        position = 0
                
                elif position == -1:  # Short position
                    if high.iloc[i] >= upper_channel.iloc[i]:
                        trades.append({
                            'type': 'BUY',
                            'price': close.iloc[i],
                            'timestamp': market_data.index[i],
                            'atr': atr.iloc[i] if not pd.isna(atr.iloc[i]) else 0.01
                        })
                        position = 0
            
            return trades
            
        except Exception as e:
            logger.warning(f"Error simulating Donchian trades: {e}")
            return []
    
    def _simulate_generic_trades(self, market_data: pd.DataFrame, config: Dict) -> List[Dict]:
        """Simulate generic momentum strategy trades"""
        try:
            close = market_data['close']
            
            # Simple momentum strategy
            momentum_period = config.get('momentum_period', 10)
            momentum = close.pct_change(momentum_period)
            
            trades = []
            position = 0
            
            momentum_threshold = config.get('momentum_threshold', 0.02)  # 2%
            
            for i in range(momentum_period, len(market_data)):
                if pd.isna(momentum.iloc[i]):
                    continue
                
                if position == 0:
                    # Buy on positive momentum
                    if momentum.iloc[i] > momentum_threshold:
                        trades.append({
                            'type': 'BUY',
                            'price': close.iloc[i],
                            'timestamp': market_data.index[i],
                            'momentum': momentum.iloc[i]
                        })
                        position = 1
                    # Sell on negative momentum
                    elif momentum.iloc[i] < -momentum_threshold:
                        trades.append({
                            'type': 'SELL',
                            'price': close.iloc[i],
                            'timestamp': market_data.index[i],
                            'momentum': momentum.iloc[i]
                        })
                        position = -1
                
                # Exit on momentum reversal
                elif position == 1 and momentum.iloc[i] < 0:
                    trades.append({
                        'type': 'SELL',
                        'price': close.iloc[i],
                        'timestamp': market_data.index[i],
                        'momentum': momentum.iloc[i]
                    })
                    position = 0
                
                elif position == -1 and momentum.iloc[i] > 0:
                    trades.append({
                        'type': 'BUY',
                        'price': close.iloc[i],
                        'timestamp': market_data.index[i],
                        'momentum': momentum.iloc[i]
                    })
                    position = 0
            
            return trades
            
        except Exception as e:
            logger.warning(f"Error simulating generic trades: {e}")
            return []
    
    def _simulate_ai_enhanced_trades(
        self,
        strategy_name: str,
        strategy_config: Dict,
        market_data: pd.DataFrame,
        consensus_result
    ) -> List[Dict[str, Any]]:
        """Simulate trades with AI enhancement"""
        try:
            # Get base trades from strategy
            base_trades = self._simulate_strategy_trades(strategy_name, strategy_config, market_data)
            
            if not base_trades:
                return []
            
            # Enhance trades based on AI consensus
            enhanced_trades = []
            
            for trade in base_trades:
                # Apply AI confidence scoring
                ai_confidence = getattr(consensus_result, 'overall_confidence', 0.7)
                
                # Filter trades based on AI confidence
                confidence_threshold = 0.6
                if ai_confidence >= confidence_threshold:
                    # Enhance trade with AI metrics
                    enhanced_trade = trade.copy()
                    enhanced_trade['ai_confidence'] = ai_confidence
                    enhanced_trade['ai_enhanced'] = True
                    enhanced_trade['consensus_level'] = getattr(consensus_result, 'consensus_level', 'unknown')
                    
                    # Adjust position size based on confidence
                    size_multiplier = min(1.5, ai_confidence / 0.5)  # Max 1.5x position size
                    enhanced_trade['size_multiplier'] = size_multiplier
                    
                    enhanced_trades.append(enhanced_trade)
            
            return enhanced_trades
            
        except Exception as e:
            logger.warning(f"Error simulating AI-enhanced trades: {e}")
            return self._simulate_strategy_trades(strategy_name, strategy_config, market_data)
    
    def _calculate_returns(self, trades: List[Dict], config: BacktestConfig) -> np.ndarray:
        """Calculate returns from trades"""
        try:
            if len(trades) < 2:
                return np.array([])
            
            returns = []
            position = 0
            entry_price = 0
            
            for trade in trades:
                if trade['type'] == 'BUY':
                    if position == 0:  # Opening long position
                        position = 1
                        entry_price = trade['price']
                    elif position == -1:  # Closing short position
                        trade_return = (entry_price - trade['price']) / entry_price
                        # Apply costs
                        trade_return -= config.commission + config.slippage
                        returns.append(trade_return)
                        position = 1
                        entry_price = trade['price']
                
                elif trade['type'] == 'SELL':
                    if position == 0:  # Opening short position
                        position = -1
                        entry_price = trade['price']
                    elif position == 1:  # Closing long position
                        trade_return = (trade['price'] - entry_price) / entry_price
                        # Apply costs
                        trade_return -= config.commission + config.slippage
                        returns.append(trade_return)
                        position = -1
                        entry_price = trade['price']
            
            return np.array(returns)
            
        except Exception as e:
            logger.warning(f"Error calculating returns: {e}")
            return np.array([])
    
    def _calculate_performance_metrics(
        self,
        returns: np.ndarray,
        trades: List[Dict],
        config: BacktestConfig
    ) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        try:
            if len(returns) == 0:
                return self._get_empty_metrics()
            
            # Basic metrics
            total_return = (np.prod(1 + returns) - 1) * 100
            annual_return = ((1 + total_return/100) ** (252/len(returns)) - 1) * 100 if len(returns) > 0 else 0
            
            # Risk metrics
            volatility = np.std(returns) * np.sqrt(252) * 100
            sharpe_ratio = (np.mean(returns) * 252 - self.risk_free_rate) / (np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
            
            # Downside deviation for Sortino ratio
            downside_returns = returns[returns < 0]
            downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = (np.mean(returns) * 252 - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
            
            # Drawdown analysis
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdown) * 100
            
            # Drawdown duration
            max_dd_duration = self._calculate_max_drawdown_duration(drawdown)
            
            # Trade statistics
            win_rate = np.mean(returns > 0) * 100 if len(returns) > 0 else 0
            avg_trade_return = np.mean(returns) * 100
            
            # Profit factor
            winning_trades = returns[returns > 0]
            losing_trades = returns[returns < 0]
            gross_profit = np.sum(winning_trades) if len(winning_trades) > 0 else 0
            gross_loss = abs(np.sum(losing_trades)) if len(losing_trades) > 0 else 0.0001  # Avoid division by zero
            profit_factor = gross_profit / gross_loss
            
            # Calmar ratio
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Value at Risk (VaR)
            var_95 = np.percentile(returns, 5) * 100
            var_99 = np.percentile(returns, 1) * 100
            
            # Conditional VaR (Expected Shortfall)
            cvar_95 = np.mean(returns[returns <= np.percentile(returns, 5)]) * 100 if len(returns) > 0 else 0
            
            return {
                'total_return': total_return,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'max_drawdown_duration': max_dd_duration,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'trades_count': len(trades),
                'avg_trade_return': avg_trade_return,
                'volatility': volatility,
                'calmar_ratio': calmar_ratio,
                'value_at_risk': var_95,
                'conditional_var': cvar_95,
                'ai_confidence_correlation': 0.0,  # Will be calculated separately for AI-enhanced
                'strategy_consistency_score': win_rate / 100 if win_rate > 0 else 0,
                'regime_adaptability': 0.7  # Default value, can be enhanced with regime analysis
            }
            
        except Exception as e:
            logger.warning(f"Error calculating performance metrics: {e}")
            return self._get_empty_metrics()
    
    def _calculate_max_drawdown_duration(self, drawdown: np.ndarray) -> int:
        """Calculate maximum drawdown duration in periods"""
        try:
            max_duration = 0
            current_duration = 0
            
            for dd in drawdown:
                if dd < 0:
                    current_duration += 1
                    max_duration = max(max_duration, current_duration)
                else:
                    current_duration = 0
            
            return max_duration
            
        except Exception:
            return 0
    
    def _calculate_ai_enhanced_metrics(
        self,
        trades: List[Dict],
        consensus_result,
        chatgpt_analysis: Dict,
        manus_analysis: Dict
    ) -> Dict[str, float]:
        """Calculate AI-specific performance metrics"""
        try:
            # Extract AI confidence scores from trades
            ai_confidences = [trade.get('ai_confidence', 0.5) for trade in trades if 'ai_confidence' in trade]
            
            # Calculate confidence correlation with trade performance
            if len(ai_confidences) >= 2:
                # Simplified correlation calculation
                confidence_correlation = np.corrcoef(ai_confidences, [1] * len(ai_confidences))[0, 1]
                if np.isnan(confidence_correlation):
                    confidence_correlation = 0.0
            else:
                confidence_correlation = 0.0
            
            # Strategy suitability score based on AI consensus
            consensus_confidence = getattr(consensus_result, 'overall_confidence', 0.5)
            agreement_score = getattr(consensus_result, 'agreement_score', 0.5)
            
            strategy_suitability = (consensus_confidence + agreement_score) / 2
            
            # AI enhancement effectiveness
            enhanced_trades = [t for t in trades if t.get('ai_enhanced', False)]
            enhancement_ratio = len(enhanced_trades) / len(trades) if trades else 0
            
            return {
                'ai_confidence_correlation': confidence_correlation,
                'strategy_suitability_score': strategy_suitability,
                'ai_enhancement_ratio': enhancement_ratio,
                'consensus_confidence': consensus_confidence,
                'ai_agreement_score': agreement_score,
                'chatgpt_confidence': chatgpt_analysis.get('confidence_score', 0.5),
                'manus_confidence': manus_analysis.get('confidence', 0.5) if isinstance(manus_analysis, dict) else 0.5
            }
            
        except Exception as e:
            logger.warning(f"Error calculating AI-enhanced metrics: {e}")
            return {
                'ai_confidence_correlation': 0.0,
                'strategy_suitability_score': 0.5,
                'ai_enhancement_ratio': 0.0,
                'consensus_confidence': 0.5,
                'ai_agreement_score': 0.5,
                'chatgpt_confidence': 0.5,
                'manus_confidence': 0.5
            }
    
    # Additional helper methods would continue here...
    # (Including parameter optimization, strategy comparison, etc.)
    
    def _get_empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics for failed backtests"""
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_duration': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'trades_count': 0,
            'avg_trade_return': 0.0,
            'volatility': 0.0,
            'calmar_ratio': 0.0,
            'value_at_risk': 0.0,
            'conditional_var': 0.0,
            'ai_confidence_correlation': 0.0,
            'strategy_consistency_score': 0.0,
            'regime_adaptability': 0.0
        }
    
    def _create_empty_results(self, error_msg: str) -> BacktestResults:
        """Create empty backtest results for errors"""
        empty_metrics = self._get_empty_metrics()
        return BacktestResults(**empty_metrics)
    
    def _create_scenario_data(self, original_data: pd.DataFrame, scenario_returns: np.ndarray) -> pd.DataFrame:
        """Create scenario price data from bootstrapped returns"""
        try:
            scenario_data = original_data.copy()
            
            # Generate new prices based on scenario returns
            initial_price = original_data['close'].iloc[0]
            new_prices = [initial_price]
            
            for i, ret in enumerate(scenario_returns[1:]):  # Skip first return
                new_price = new_prices[-1] * (1 + ret)
                new_prices.append(new_price)
            
            # Update all OHLC based on new closes (simplified)
            scenario_data['close'] = new_prices[:len(scenario_data)]
            scenario_data['open'] = scenario_data['close'].shift(1).fillna(scenario_data['close'].iloc[0])
            scenario_data['high'] = scenario_data[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.01, len(scenario_data)))
            scenario_data['low'] = scenario_data[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.01, len(scenario_data)))
            
            return scenario_data
            
        except Exception as e:
            logger.warning(f"Error creating scenario data: {e}")
            return original_data
    
    # Placeholder methods for remaining functionality
    def _generate_parameter_combinations(self, parameter_ranges: Dict) -> List[Dict]:
        """Generate parameter combinations for optimization"""
        # Simplified implementation - would be more sophisticated in practice
        combinations = []
        
        # For each parameter, create a range of values
        param_names = list(parameter_ranges.keys())
        if not param_names:
            return [{}]
        
        # Simple grid search (limited to prevent excessive combinations)
        max_combinations = 50
        
        import itertools
        
        param_values = []
        for param_name, (min_val, max_val, step) in parameter_ranges.items():
            values = np.arange(min_val, max_val + step, step)
            param_values.append([(param_name, val) for val in values[:10]])  # Limit values per parameter
        
        # Generate combinations
        for combination in itertools.product(*param_values):
            if len(combinations) >= max_combinations:
                break
            param_dict = dict(combination)
            combinations.append(param_dict)
        
        return combinations
    
    def _test_parameter_combination(self, symbol: str, strategy_name: str, 
                                  params: Dict, market_data: pd.DataFrame) -> Dict:
        """Test a single parameter combination"""
        try:
            # Run simplified backtest with parameters
            trades = self._simulate_strategy_trades(strategy_name, params, market_data)
            
            if not trades:
                return {'error': 'No trades generated'}
            
            returns = self._calculate_returns(trades, self.default_config)
            
            if len(returns) == 0:
                return {'error': 'No valid returns'}
            
            # Calculate key metrics
            total_return = (np.prod(1 + returns) - 1) * 100
            sharpe_ratio = (np.mean(returns) * 252) / (np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
            max_drawdown = self._calculate_simple_max_drawdown(returns)
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'trades_count': len(trades)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_simple_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns"""
        try:
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            return np.min(drawdown) * 100
        except Exception:
            return 0.0
    
    async def _validate_optimized_parameters(self, symbol: str, strategy_name: str, 
                                           params: Dict, market_data: pd.DataFrame) -> Dict:
        """Validate optimized parameters using AI"""
        try:
            # Get AI validation of the optimized parameters
            validation_result = await self.ai_consensus.validate_strategy_recommendation(
                symbol, strategy_name, market_data, params
            )
            
            return {
                'validation_passed': validation_result.get('recommendation') == 'strongly_recommended',
                'confidence': validation_result.get('confidence', 0.5),
                'ai_feedback': validation_result.get('consensus_validation', {})
            }
            
        except Exception as e:
            return {'validation_passed': False, 'error': str(e)}
    
    def _generate_optimization_report(self, optimization_results: List[Dict], 
                                    best_result: Dict, metric: str) -> Dict:
        """Generate parameter optimization report"""
        try:
            if not optimization_results:
                return {'error': 'No optimization results'}
            
            # Calculate optimization statistics
            metric_values = [r.get(metric, 0) for r in optimization_results]
            
            report = {
                'total_combinations_tested': len(optimization_results),
                'best_metric_value': best_result.get(metric, 0),
                'average_metric_value': np.mean(metric_values),
                'metric_standard_deviation': np.std(metric_values),
                'improvement_percentage': ((best_result.get(metric, 0) - np.mean(metric_values)) / 
                                         np.mean(metric_values) * 100) if np.mean(metric_values) != 0 else 0,
                'optimization_metric': metric
            }
            
            return report
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_performance_improvement(self, optimization_results: List[Dict]) -> Dict:
        """Calculate performance improvement from optimization"""
        try:
            if len(optimization_results) < 2:
                return {'improvement': 0, 'confidence': 'low'}
            
            # Compare best vs average performance
            returns = [r.get('total_return', 0) for r in optimization_results]
            sharpes = [r.get('sharpe_ratio', 0) for r in optimization_results if r.get('sharpe_ratio', 0) != 0]
            
            best_return = max(returns) if returns else 0
            avg_return = np.mean(returns) if returns else 0
            
            improvement = ((best_return - avg_return) / avg_return * 100) if avg_return != 0 else 0
            
            return {
                'return_improvement': improvement,
                'best_return': best_return,
                'average_return': avg_return,
                'confidence': 'high' if len(optimization_results) > 20 else 'medium'
            }
            
        except Exception as e:
            return {'improvement': 0, 'error': str(e)}
    
    def _generate_comprehensive_analysis(self, results: Dict, symbol: str, strategy_name: str) -> Dict:
        """Generate comprehensive analysis of all backtest results"""
        try:
            analysis = {
                'strategy_viability': 'unknown',
                'risk_assessment': 'medium',
                'market_suitability': 'unknown',
                'ai_consensus': 'neutral',
                'key_findings': [],
                'recommendations': []
            }
            
            # Analyze simple backtest results
            if 'simple' in results and 'error' not in results['simple']:
                simple_results = results['simple']
                if hasattr(simple_results, 'sharpe_ratio') and simple_results.sharpe_ratio > 1.0:
                    analysis['strategy_viability'] = 'good'
                    analysis['key_findings'].append('Strategy shows good risk-adjusted returns')
                elif hasattr(simple_results, 'total_return') and simple_results.total_return > 0:
                    analysis['strategy_viability'] = 'marginal'
                else:
                    analysis['strategy_viability'] = 'poor'
            
            # Analyze Monte Carlo results
            if 'monte_carlo' in results and 'error' not in results['monte_carlo']:
                mc_results = results['monte_carlo']
                prob_positive = mc_results.get('probability_positive', 0)
                if prob_positive > 0.6:
                    analysis['key_findings'].append(f'Monte Carlo shows {prob_positive:.1%} probability of positive returns')
                    analysis['recommendations'].append('Strategy shows consistent performance across scenarios')
            
            # Analyze AI-enhanced results
            if 'ai_enhanced' in results and 'error' not in results['ai_enhanced']:
                ai_results = results['ai_enhanced']
                ai_confidence = ai_results.get('ai_consensus_score', 0.5)
                if ai_confidence > 0.7:
                    analysis['ai_consensus'] = 'positive'
                    analysis['recommendations'].append('AI systems show high confidence in strategy')
                elif ai_confidence < 0.4:
                    analysis['ai_consensus'] = 'negative'
                    analysis['recommendations'].append('AI systems suggest caution with this strategy')
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Error generating comprehensive analysis: {e}")
            return {'error': str(e)}
    
    def _calculate_overall_strategy_score(self, results: Dict) -> float:
        """Calculate overall strategy score from all backtest results"""
        try:
            scores = []
            weights = {
                'simple': 0.3,
                'monte_carlo': 0.2,
                'walk_forward': 0.25,
                'ai_enhanced': 0.25
            }
            
            for backtest_type, weight in weights.items():
                if backtest_type in results and 'error' not in results[backtest_type]:
                    result = results[backtest_type]
                    
                    if backtest_type == 'simple':
                        # Score based on Sharpe ratio and return
                        if hasattr(result, 'sharpe_ratio') and hasattr(result, 'total_return'):
                            score = min(100, max(0, result.sharpe_ratio * 30 + result.total_return))
                        else:
                            score = 50
                    elif backtest_type == 'monte_carlo':
                        # Score based on probability of positive returns
                        score = result.get('probability_positive', 0.5) * 100
                    elif backtest_type == 'walk_forward':
                        # Score based on consistency
                        score = result.get('consistent_performance', 0.5) * 100
                    elif backtest_type == 'ai_enhanced':
                        # Score based on AI confidence
                        score = result.get('ai_consensus_score', 0.5) * 100
                    else:
                        score = 50
                    
                    scores.append(score * weight)
            
            return sum(scores) if scores else 50.0
            
        except Exception as e:
            logger.warning(f"Error calculating overall strategy score: {e}")
            return 50.0
    
    def _generate_strategy_recommendation(self, overall_score: float, results: Dict) -> str:
        """Generate strategy recommendation based on overall score"""
        if overall_score >= 80:
            return "Highly Recommended - Strong performance across all metrics"
        elif overall_score >= 65:
            return "Recommended - Good performance with acceptable risk"
        elif overall_score >= 50:
            return "Conditional - Consider with additional risk management"
        elif overall_score >= 35:
            return "Not Recommended - Poor risk-adjusted performance"
        else:
            return "Strongly Not Recommended - High risk of losses"
    
    def _generate_risk_assessment(self, results: Dict) -> Dict:
        """Generate comprehensive risk assessment"""
        try:
            risk_assessment = {
                'overall_risk': 'medium',
                'max_drawdown_risk': 'medium',
                'volatility_risk': 'medium',
                'model_risk': 'medium',
                'key_risks': [],
                'mitigation_suggestions': []
            }
            
            # Analyze drawdown risk
            if 'simple' in results and hasattr(results['simple'], 'max_drawdown'):
                max_dd = abs(results['simple'].max_drawdown)
                if max_dd > 20:
                    risk_assessment['max_drawdown_risk'] = 'high'
                    risk_assessment['key_risks'].append(f'High maximum drawdown: {max_dd:.1f}%')
                elif max_dd < 5:
                    risk_assessment['max_drawdown_risk'] = 'low'
            
            # Analyze volatility risk
            if 'simple' in results and hasattr(results['simple'], 'volatility'):
                volatility = results['simple'].volatility
                if volatility > 25:
                    risk_assessment['volatility_risk'] = 'high'
                    risk_assessment['key_risks'].append(f'High volatility: {volatility:.1f}%')
                elif volatility < 10:
                    risk_assessment['volatility_risk'] = 'low'
            
            # Add mitigation suggestions
            if risk_assessment['max_drawdown_risk'] == 'high':
                risk_assessment['mitigation_suggestions'].append('Consider tighter stop losses')
            if risk_assessment['volatility_risk'] == 'high':
                risk_assessment['mitigation_suggestions'].append('Reduce position sizes during high volatility')
            
            return risk_assessment
            
        except Exception as e:
            logger.warning(f"Error generating risk assessment: {e}")
            return {'overall_risk': 'unknown', 'error': str(e)}
    
    def _generate_performance_summary(self, results: Dict) -> Dict:
        """Generate performance summary across all backtests"""
        try:
            summary = {
                'total_return_range': 'N/A',
                'sharpe_ratio_range': 'N/A',
                'consistency_score': 'N/A',
                'ai_confidence': 'N/A'
            }
            
            # Collect performance metrics
            returns = []
            sharpes = []
            
            if 'simple' in results and hasattr(results['simple'], 'total_return'):
                returns.append(results['simple'].total_return)
                if hasattr(results['simple'], 'sharpe_ratio'):
                    sharpes.append(results['simple'].sharpe_ratio)
            
            if 'monte_carlo' in results:
                mc_return = results['monte_carlo'].get('mean_return', 0)
                returns.append(mc_return)
            
            # Calculate ranges
            if returns:
                summary['total_return_range'] = f"{min(returns):.1f}% to {max(returns):.1f}%"
            
            if sharpes:
                summary['sharpe_ratio_range'] = f"{min(sharpes):.2f} to {max(sharpes):.2f}"
            
            # AI confidence
            if 'ai_enhanced' in results:
                ai_conf = results['ai_enhanced'].get('ai_consensus_score', 0.5)
                summary['ai_confidence'] = f"{ai_conf:.1%}"
            
            return summary
            
        except Exception as e:
            logger.warning(f"Error generating performance summary: {e}")
            return {'error': str(e)}
    
    def _generate_strategy_comparison(self, strategy_results: Dict, metrics: List[str]) -> Dict:
        """Generate comparison analysis between strategies"""
        try:
            comparison = {
                'performance_rankings': {},
                'risk_rankings': {},
                'best_overall_strategy': None,
                'strategy_strengths': {},
                'strategy_weaknesses': {}
            }
            
            # Rank strategies by each metric
            for metric in metrics:
                metric_scores = {}
                
                for strategy_name, results in strategy_results.items():
                    if 'error' not in results and 'backtest_results' in results:
                        simple_results = results['backtest_results'].get('simple')
                        if simple_results and hasattr(simple_results, metric):
                            metric_scores[strategy_name] = getattr(simple_results, metric)
                
                # Sort by metric (higher is better for most metrics except drawdown)
                reverse_sort = metric != 'max_drawdown'
                sorted_strategies = sorted(
                    metric_scores.items(),
                    key=lambda x: x[1],
                    reverse=reverse_sort
                )
                
                comparison['performance_rankings'][metric] = [
                    {'strategy': name, 'value': value} for name, value in sorted_strategies
                ]
            
            # Determine best overall strategy
            if comparison['performance_rankings']:
                # Simple scoring: average rank across metrics
                strategy_scores = {}
                for strategy_name in strategy_results.keys():
                    if 'error' not in strategy_results[strategy_name]:
                        scores = []
                        for metric_ranking in comparison['performance_rankings'].values():
                            for i, entry in enumerate(metric_ranking):
                                if entry['strategy'] == strategy_name:
                                    scores.append(len(metric_ranking) - i)  # Higher rank = higher score
                                    break
                        
                        if scores:
                            strategy_scores[strategy_name] = np.mean(scores)
                
                if strategy_scores:
                    comparison['best_overall_strategy'] = max(strategy_scores.items(), key=lambda x: x[1])[0]
            
            return comparison
            
        except Exception as e:
            logger.warning(f"Error generating strategy comparison: {e}")
            return {'error': str(e)}
    
    async def _get_ai_strategy_recommendation(self, symbol: str, strategy_results: Dict, 
                                            market_data: pd.DataFrame) -> Dict:
        """Get AI recommendation for best strategy"""
        try:
            # Extract strategy names and basic performance
            strategy_info = []
            
            for strategy_name, results in strategy_results.items():
                if 'error' not in results and 'backtest_results' in results:
                    simple_results = results['backtest_results'].get('simple')
                    if simple_results:
                        strategy_info.append({
                            'name': strategy_name,
                            'total_return': getattr(simple_results, 'total_return', 0),
                            'sharpe_ratio': getattr(simple_results, 'sharpe_ratio', 0),
                            'max_drawdown': getattr(simple_results, 'max_drawdown', 0)
                        })
            
            if not strategy_info:
                return {'recommendation': 'No viable strategies found'}
            
            # Get AI consensus on best strategy
            strategy_names = [s['name'] for s in strategy_info]
            consensus_result = await self.ai_consensus.generate_consensus_recommendation(
                symbol, market_data, strategy_names
            )
            
            return {
                'ai_recommended_strategy': consensus_result.recommended_strategies[0]['name'] if consensus_result.recommended_strategies else None,
                'confidence': consensus_result.overall_confidence,
                'reasoning': consensus_result.reasoning,
                'consensus_level': consensus_result.consensus_level.value
            }
            
        except Exception as e:
            logger.warning(f"Error getting AI strategy recommendation: {e}")
            return {'recommendation': 'AI analysis unavailable', 'error': str(e)}

# Export main class
__all__ = ['AdvancedBacktester', 'BacktestConfig', 'BacktestResults', 'BacktestType']