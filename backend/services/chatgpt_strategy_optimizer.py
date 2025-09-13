"""
ChatGPT Strategy Optimizer Service
Advanced AI-powered trading strategy optimization using OpenAI's GPT-5 model
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from openai import OpenAI

from ..logs.logger import get_logger
from ..signals.utils import calculate_atr, calculate_volatility

# the newest OpenAI model is "gpt-5" which was released August 7, 2025.
# do not change this unless explicitly requested by the user

logger = get_logger(__name__)

class ChatGPTStrategyOptimizer:
    """
    Advanced ChatGPT-powered strategy optimizer for intelligent trading decisions
    Uses GPT-5 for sophisticated market analysis, strategy optimization, and risk assessment
    """
    
    def __init__(self):
        self.name = "ChatGPT Strategy Optimizer"
        
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.openai_client = OpenAI(api_key=api_key)
        
        # Model configuration - Using GPT-5 (latest model)
        self.model = "gpt-5"
        self.max_tokens = 4096
        self.temperature = 0.2  # Lower temperature for more deterministic financial analysis
        
        # Trading strategy knowledge base
        self.strategy_descriptions = {
            'ema_rsi': {
                'name': 'EMA + RSI Momentum',
                'description': 'Combines Exponential Moving Average crossovers with RSI momentum for trend following',
                'best_conditions': 'Strong trending markets with clear directional momentum',
                'risk_profile': 'Medium risk, good for trending periods',
                'timeframes': ['1H', '4H', '1D']
            },
            'donchian_atr': {
                'name': 'Donchian Breakout + ATR',
                'description': 'Donchian channel breakouts with ATR-based position sizing and stops',
                'best_conditions': 'High volatility breakout scenarios and strong trend initiation',
                'risk_profile': 'Higher risk, excellent for capturing major moves',
                'timeframes': ['4H', '1D']
            },
            'meanrev_bb': {
                'name': 'Mean Reversion + Bollinger Bands',
                'description': 'Mean reversion strategy using Bollinger Band extremes and oversold/overbought conditions',
                'best_conditions': 'Range-bound markets with clear support/resistance levels',
                'risk_profile': 'Lower risk, consistent returns in sideways markets',
                'timeframes': ['1H', '4H']
            },
            'macd_strategy': {
                'name': 'MACD Signal Strategy',
                'description': 'MACD line and signal line crossovers with histogram momentum confirmation',
                'best_conditions': 'Medium-term trend changes and momentum shifts',
                'risk_profile': 'Medium risk, good for trend transition periods',
                'timeframes': ['4H', '1D']
            },
            'stochastic': {
                'name': 'Stochastic Oscillator',
                'description': 'Stochastic %K and %D crossovers for overbought/oversold conditions',
                'best_conditions': 'Range-bound markets and short-term reversal points',
                'risk_profile': 'Lower risk, high frequency signals',
                'timeframes': ['1H', '4H']
            },
            'rsi_divergence': {
                'name': 'RSI Divergence Strategy',
                'description': 'Identifies price-RSI divergences for potential reversal signals',
                'best_conditions': 'Market extremes and potential reversal points',
                'risk_profile': 'Medium risk, excellent for catching reversals',
                'timeframes': ['4H', '1D']
            },
            'fibonacci': {
                'name': 'Fibonacci Retracement Strategy',
                'description': 'Uses Fibonacci levels for entry/exit points in trending markets',
                'best_conditions': 'Strong trends with healthy retracements',
                'risk_profile': 'Medium risk, precision entry timing',
                'timeframes': ['4H', '1D']
            }
        }
        
        # Market condition templates for GPT analysis
        self.market_analysis_template = """
        As an expert quantitative analyst and trading strategist, analyze the following market data and provide sophisticated trading insights:

        MARKET DATA ANALYSIS:
        Symbol: {symbol}
        Current Price: {current_price}
        Market Regime: {regime}
        Volatility (ATR%): {volatility_pct:.3f}%
        Price Change (24h): {price_change_pct:.2f}%
        
        TECHNICAL INDICATORS:
        RSI (14): {rsi:.2f}
        MACD: {macd:.4f}
        Bollinger Band Position: {bb_position}
        Moving Average Trend: {ma_trend}
        
        STRATEGY OPTIONS:
        {strategy_options}
        
        ANALYSIS REQUIREMENTS:
        1. Market Condition Assessment: Analyze current market structure, volatility regime, and trend strength
        2. Strategy Ranking: Rank the provided strategies from best to worst for current conditions
        3. Risk Assessment: Evaluate position sizing and risk management recommendations
        4. Entry/Exit Timing: Suggest optimal entry conditions and exit strategies
        5. Market Outlook: Provide short-term (1-4 hours) and medium-term (1-7 days) market outlook
        
        Respond in JSON format with specific, actionable recommendations based on quantitative analysis.
        """
        
        logger.info(f"ChatGPT Strategy Optimizer initialized with model: {self.model}")
    
    async def analyze_market_conditions(self, symbol: str, market_data: pd.DataFrame, 
                                      available_strategies: List[str]) -> Dict[str, Any]:
        """
        Comprehensive market analysis using ChatGPT's advanced reasoning capabilities
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD', 'BTCUSD')
            market_data: OHLC price data
            available_strategies: List of strategy names to analyze
            
        Returns:
            Dict with detailed market analysis and strategy recommendations
        """
        try:
            # Calculate technical indicators for GPT analysis
            tech_indicators = self._calculate_technical_indicators(market_data)
            
            # Determine market regime and conditions
            market_conditions = self._assess_market_conditions(market_data, symbol)
            
            # Prepare strategy descriptions for analysis
            strategy_options = self._format_strategy_options(available_strategies)
            
            # Create comprehensive prompt for GPT analysis
            prompt = self.market_analysis_template.format(
                symbol=symbol,
                current_price=market_data['close'].iloc[-1],
                regime=market_conditions['regime'],
                volatility_pct=market_conditions['volatility_pct'],
                price_change_pct=market_conditions['price_change_pct'],
                rsi=tech_indicators['rsi'],
                macd=tech_indicators['macd'],
                bb_position=tech_indicators['bb_position'],
                ma_trend=tech_indicators['ma_trend'],
                strategy_options=strategy_options
            )
            
            # Get GPT-5 analysis
            gpt_analysis = await self._get_gpt_analysis(prompt, analysis_type="market_conditions")
            
            # Enhance with quantitative validation
            enhanced_analysis = self._enhance_with_quantitative_validation(
                gpt_analysis, market_data, tech_indicators
            )
            
            result = {
                'symbol': symbol,
                'timestamp': datetime.utcnow().isoformat(),
                'market_conditions': market_conditions,
                'technical_indicators': tech_indicators,
                'gpt_analysis': enhanced_analysis,
                'confidence_score': self._calculate_analysis_confidence(enhanced_analysis, tech_indicators),
                'recommended_strategies': enhanced_analysis.get('strategy_ranking', []),
                'risk_assessment': enhanced_analysis.get('risk_assessment', {}),
                'market_outlook': enhanced_analysis.get('market_outlook', {})
            }
            
            logger.info(f"ChatGPT market analysis completed for {symbol} with confidence: "
                       f"{result['confidence_score']:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in ChatGPT market analysis for {symbol}: {e}")
            return self._fallback_analysis(symbol, available_strategies)
    
    async def optimize_strategy_parameters(self, symbol: str, strategy_name: str, 
                                         market_data: pd.DataFrame, 
                                         current_params: Dict) -> Dict[str, Any]:
        """
        AI-powered strategy parameter optimization using ChatGPT's reasoning
        
        Args:
            symbol: Trading symbol
            strategy_name: Name of strategy to optimize
            market_data: Historical price data
            current_params: Current strategy parameters
            
        Returns:
            Dict with optimized parameters and performance projections
        """
        try:
            # Calculate strategy performance with current parameters
            current_performance = self._calculate_strategy_performance(
                market_data, strategy_name, current_params
            )
            
            # Market condition analysis for optimization context
            market_conditions = self._assess_market_conditions(market_data, symbol)
            
            # Create optimization prompt
            optimization_prompt = f"""
            As a quantitative trading expert specializing in parameter optimization, analyze and optimize the following strategy:

            STRATEGY: {self.strategy_descriptions.get(strategy_name, {}).get('name', strategy_name)}
            SYMBOL: {symbol}
            
            CURRENT PARAMETERS:
            {json.dumps(current_params, indent=2)}
            
            CURRENT PERFORMANCE METRICS:
            Total Return: {current_performance.get('total_return', 0):.2f}%
            Sharpe Ratio: {current_performance.get('sharpe_ratio', 0):.3f}
            Maximum Drawdown: {current_performance.get('max_drawdown', 0):.2f}%
            Win Rate: {current_performance.get('win_rate', 0):.1f}%
            
            MARKET CONDITIONS:
            Regime: {market_conditions['regime']}
            Volatility: {market_conditions['volatility_pct']:.3f}%
            Trend Strength: {market_conditions.get('trend_strength', 'Unknown')}
            
            OPTIMIZATION REQUIREMENTS:
            1. Parameter Recommendations: Suggest optimal parameter values based on current market conditions
            2. Risk-Return Trade-offs: Analyze how parameter changes affect risk-return profile
            3. Market Adaptability: Ensure parameters work across different market regimes
            4. Backtesting Insights: Provide rationale for each parameter adjustment
            5. Performance Projections: Estimate expected performance improvements
            
            Respond in JSON format with specific parameter recommendations and detailed rationale.
            """
            
            # Get GPT optimization recommendations
            optimization_result = await self._get_gpt_analysis(
                optimization_prompt, analysis_type="parameter_optimization"
            )
            
            # Validate and enhance recommendations
            validated_result = self._validate_parameter_recommendations(
                optimization_result, current_params, strategy_name
            )
            
            result = {
                'symbol': symbol,
                'strategy_name': strategy_name,
                'timestamp': datetime.utcnow().isoformat(),
                'current_performance': current_performance,
                'optimization_recommendations': validated_result,
                'market_context': market_conditions,
                'confidence_score': validated_result.get('confidence', 0.7)
            }
            
            logger.info(f"Strategy parameter optimization completed for {symbol} {strategy_name}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error optimizing strategy parameters for {symbol} {strategy_name}: {e}")
            return {'error': str(e), 'fallback_params': current_params}
    
    async def assess_market_sentiment(self, symbol: str, news_data: Optional[str] = None,
                                    market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Advanced market sentiment analysis using ChatGPT's language understanding
        
        Args:
            symbol: Trading symbol
            news_data: Optional news/text data for sentiment analysis
            market_data: Optional price data for technical sentiment
            
        Returns:
            Dict with comprehensive sentiment analysis
        """
        try:
            # Create sentiment analysis prompt
            sentiment_prompt = f"""
            As a financial markets expert with deep knowledge of {symbol} trading dynamics, 
            provide a comprehensive sentiment analysis:

            SYMBOL: {symbol}
            ANALYSIS DATE: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
            
            """
            
            # Add news sentiment if available
            if news_data:
                sentiment_prompt += f"""
                NEWS/MARKET COMMENTARY:
                {news_data[:2000]}  # Limit to avoid token limits
                
                """
            
            # Add technical sentiment if market data available
            if market_data is not None:
                tech_sentiment = self._calculate_technical_sentiment(market_data)
                sentiment_prompt += f"""
                TECHNICAL SENTIMENT INDICATORS:
                Price Momentum (14d): {tech_sentiment.get('momentum_14d', 0):.2f}%
                Volume Trend: {tech_sentiment.get('volume_trend', 'Unknown')}
                Support/Resistance: {tech_sentiment.get('support_resistance', 'Unknown')}
                Volatility Regime: {tech_sentiment.get('volatility_regime', 'Unknown')}
                
                """
            
            sentiment_prompt += """
            SENTIMENT ANALYSIS REQUIREMENTS:
            1. Overall Sentiment: Bullish, Bearish, or Neutral with confidence score (0-1)
            2. Sentiment Drivers: Key factors influencing current sentiment
            3. Sentiment Shifts: Potential catalysts for sentiment changes
            4. Trading Implications: How sentiment should influence strategy selection
            5. Time Horizon: Short-term (hours) vs medium-term (days) sentiment outlook
            
            Focus on actionable insights for trading decisions. Respond in JSON format.
            """
            
            # Get GPT sentiment analysis
            sentiment_analysis = await self._get_gpt_analysis(
                sentiment_prompt, analysis_type="sentiment_analysis"
            )
            
            result = {
                'symbol': symbol,
                'timestamp': datetime.utcnow().isoformat(),
                'sentiment_analysis': sentiment_analysis,
                'confidence_score': sentiment_analysis.get('overall_confidence', 0.5),
                'sentiment_direction': sentiment_analysis.get('overall_sentiment', 'neutral'),
                'trading_implications': sentiment_analysis.get('trading_implications', {})
            }
            
            logger.info(f"Market sentiment analysis completed for {symbol}: "
                       f"{result['sentiment_direction']} (confidence: {result['confidence_score']:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis for {symbol}: {e}")
            return {
                'symbol': symbol,
                'sentiment_direction': 'neutral',
                'confidence_score': 0.0,
                'error': str(e)
            }
    
    async def _get_gpt_analysis(self, prompt: str, analysis_type: str) -> Dict[str, Any]:
        """Get analysis from GPT-5 with proper error handling and JSON parsing"""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a world-class quantitative analyst and trading expert with "
                                 "deep knowledge of financial markets, technical analysis, and risk management. "
                                 "Provide precise, actionable analysis in JSON format. Always include confidence "
                                 "scores and detailed reasoning for your recommendations."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Add metadata
            result['analysis_type'] = analysis_type
            result['model_used'] = self.model
            result['analysis_timestamp'] = datetime.utcnow().isoformat()
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse GPT JSON response for {analysis_type}: {e}")
            return {'error': 'JSON parsing failed', 'raw_response': response.choices[0].message.content}
        
        except Exception as e:
            logger.error(f"GPT API error for {analysis_type}: {e}")
            return {'error': str(e), 'analysis_type': analysis_type}
    
    def _calculate_technical_indicators(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators for GPT analysis"""
        try:
            # Basic indicators
            close = market_data['close']
            high = market_data['high']
            low = market_data['low']
            
            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # MACD
            ema_12 = close.ewm(span=12).mean()
            ema_26 = close.ewm(span=26).mean()
            macd = ema_12 - ema_26
            
            # Bollinger Bands
            sma_20 = close.rolling(window=20).mean()
            std_20 = close.rolling(window=20).std()
            bb_upper = sma_20 + (std_20 * 2)
            bb_lower = sma_20 - (std_20 * 2)
            bb_position = (close.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
            
            # Moving average trend
            ma_50 = close.rolling(window=50).mean()
            ma_trend = "Bullish" if close.iloc[-1] > ma_50.iloc[-1] else "Bearish"
            
            return {
                'rsi': rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0,
                'macd': macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else 0.0,
                'bb_position': f"{bb_position:.2f}" if not pd.isna(bb_position) else "0.50",
                'ma_trend': ma_trend
            }
            
        except Exception as e:
            logger.warning(f"Error calculating technical indicators: {e}")
            return {
                'rsi': 50.0,
                'macd': 0.0,
                'bb_position': "0.50",
                'ma_trend': "Neutral"
            }
    
    def _assess_market_conditions(self, market_data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Assess current market conditions"""
        try:
            close = market_data['close']
            high = market_data['high']
            low = market_data['low']
            
            # Calculate volatility
            atr = calculate_atr(high, low, close, period=14)
            current_atr = atr.iloc[-1] if len(atr) > 0 else 0.01
            volatility_pct = (current_atr / close.iloc[-1]) * 100
            
            # Price change calculation
            price_change_pct = ((close.iloc[-1] - close.iloc[-24]) / close.iloc[-24]) * 100 if len(close) >= 24 else 0
            
            # Simple regime detection
            ma_20 = close.rolling(window=20).mean()
            ma_50 = close.rolling(window=50).mean()
            
            if close.iloc[-1] > ma_20.iloc[-1] > ma_50.iloc[-1]:
                regime = "TRENDING_UP"
            elif close.iloc[-1] < ma_20.iloc[-1] < ma_50.iloc[-1]:
                regime = "TRENDING_DOWN"
            else:
                regime = "RANGING"
            
            return {
                'regime': regime,
                'volatility_pct': volatility_pct,
                'price_change_pct': price_change_pct,
                'trend_strength': 'Strong' if abs(price_change_pct) > 2 else 'Weak'
            }
            
        except Exception as e:
            logger.warning(f"Error assessing market conditions: {e}")
            return {
                'regime': 'UNKNOWN',
                'volatility_pct': 1.0,
                'price_change_pct': 0.0,
                'trend_strength': 'Unknown'
            }
    
    def _format_strategy_options(self, available_strategies: List[str]) -> str:
        """Format strategy descriptions for GPT analysis"""
        formatted_strategies = []
        
        for strategy in available_strategies:
            if strategy in self.strategy_descriptions:
                desc = self.strategy_descriptions[strategy]
                formatted_strategies.append(
                    f"• {desc['name']}: {desc['description']}\n"
                    f"  Best for: {desc['best_conditions']}\n"
                    f"  Risk Profile: {desc['risk_profile']}"
                )
        
        return "\n\n".join(formatted_strategies)
    
    def _enhance_with_quantitative_validation(self, gpt_analysis: Dict, 
                                            market_data: pd.DataFrame, 
                                            tech_indicators: Dict) -> Dict[str, Any]:
        """Enhance GPT analysis with quantitative validation"""
        try:
            # Add quantitative validation scores
            validation_scores = self._calculate_validation_scores(market_data, tech_indicators)
            
            # Enhance strategy rankings with quantitative metrics
            if 'strategy_ranking' in gpt_analysis:
                enhanced_rankings = []
                for strategy in gpt_analysis['strategy_ranking']:
                    if isinstance(strategy, dict):
                        strategy['quantitative_score'] = validation_scores.get(
                            strategy.get('name', ''), 0.5
                        )
                        enhanced_rankings.append(strategy)
                gpt_analysis['strategy_ranking'] = enhanced_rankings
            
            # Add validation metadata
            gpt_analysis['quantitative_validation'] = validation_scores
            gpt_analysis['enhanced_by_quant'] = True
            
            return gpt_analysis
            
        except Exception as e:
            logger.warning(f"Error enhancing with quantitative validation: {e}")
            return gpt_analysis
    
    def _calculate_validation_scores(self, market_data: pd.DataFrame, 
                                   tech_indicators: Dict) -> Dict[str, float]:
        """Calculate quantitative validation scores for strategies"""
        scores = {}
        
        try:
            # RSI-based validation
            rsi = tech_indicators.get('rsi', 50)
            
            # Score strategies based on RSI conditions
            if rsi < 30:  # Oversold
                scores['meanrev_bb'] = 0.8
                scores['stochastic'] = 0.7
                scores['rsi_divergence'] = 0.75
            elif rsi > 70:  # Overbought
                scores['meanrev_bb'] = 0.8
                scores['stochastic'] = 0.7
                scores['rsi_divergence'] = 0.75
            else:  # Neutral RSI
                scores['ema_rsi'] = 0.7
                scores['macd_strategy'] = 0.6
                scores['donchian_atr'] = 0.65
            
            # Default scores for all strategies
            for strategy in self.strategy_descriptions.keys():
                if strategy not in scores:
                    scores[strategy] = 0.5
            
            return scores
            
        except Exception as e:
            logger.warning(f"Error calculating validation scores: {e}")
            return {strategy: 0.5 for strategy in self.strategy_descriptions.keys()}
    
    def _calculate_analysis_confidence(self, analysis: Dict, tech_indicators: Dict) -> float:
        """Calculate overall confidence score for the analysis"""
        try:
            base_confidence = analysis.get('confidence', 0.5)
            
            # Adjust based on data quality
            rsi = tech_indicators.get('rsi', 50)
            if 20 <= rsi <= 80:  # Normal RSI range
                confidence_adjustment = 0.1
            else:
                confidence_adjustment = -0.1
            
            final_confidence = max(0.0, min(1.0, base_confidence + confidence_adjustment))
            
            return final_confidence
            
        except Exception:
            return 0.5
    
    def _calculate_strategy_performance(self, market_data: pd.DataFrame, 
                                      strategy_name: str, params: Dict) -> Dict[str, float]:
        """Calculate basic strategy performance metrics"""
        try:
            # Simple performance calculation (placeholder for actual backtesting)
            returns = market_data['close'].pct_change().dropna()
            
            # Basic metrics
            total_return = (returns.sum()) * 100
            sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
            max_drawdown = (returns.cumsum() - returns.cumsum().cummax()).min() * 100
            win_rate = (returns > 0).mean() * 100
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': abs(max_drawdown),
                'win_rate': win_rate
            }
            
        except Exception as e:
            logger.warning(f"Error calculating strategy performance: {e}")
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 50.0
            }
    
    def _validate_parameter_recommendations(self, optimization_result: Dict, 
                                          current_params: Dict, 
                                          strategy_name: str) -> Dict[str, Any]:
        """Validate and sanitize parameter recommendations"""
        try:
            # Ensure recommended parameters are within reasonable bounds
            validated_result = optimization_result.copy()
            
            # Add validation metadata
            validated_result['validation_passed'] = True
            validated_result['validation_notes'] = []
            
            # Basic validation for common parameters
            if 'recommended_parameters' in validated_result:
                params = validated_result['recommended_parameters']
                
                # Validate common parameter ranges
                if 'period' in params:
                    if not (5 <= params['period'] <= 200):
                        params['period'] = max(5, min(200, params['period']))
                        validated_result['validation_notes'].append("Period adjusted to valid range")
                
                if 'confidence_threshold' in params:
                    if not (0.5 <= params['confidence_threshold'] <= 0.95):
                        params['confidence_threshold'] = max(0.5, min(0.95, params['confidence_threshold']))
                        validated_result['validation_notes'].append("Confidence threshold adjusted")
            
            return validated_result
            
        except Exception as e:
            logger.warning(f"Error validating parameter recommendations: {e}")
            return {
                'recommended_parameters': current_params,
                'validation_passed': False,
                'error': str(e)
            }
    
    def _calculate_technical_sentiment(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical sentiment indicators"""
        try:
            close = market_data['close']
            volume = market_data.get('volume', pd.Series([1000] * len(close)))
            
            # Momentum calculation
            momentum_14d = ((close.iloc[-1] - close.iloc[-14]) / close.iloc[-14]) * 100 if len(close) >= 14 else 0
            
            # Volume trend (simplified)
            recent_volume = volume.iloc[-5:].mean() if len(volume) >= 5 else 1000
            older_volume = volume.iloc[-15:-5].mean() if len(volume) >= 15 else 1000
            volume_trend = "Increasing" if recent_volume > older_volume else "Decreasing"
            
            # Support/Resistance (simplified)
            recent_high = close.iloc[-20:].max() if len(close) >= 20 else close.iloc[-1]
            recent_low = close.iloc[-20:].min() if len(close) >= 20 else close.iloc[-1]
            current_position = (close.iloc[-1] - recent_low) / (recent_high - recent_low) if recent_high != recent_low else 0.5
            
            if current_position > 0.8:
                support_resistance = "Near resistance"
            elif current_position < 0.2:
                support_resistance = "Near support"
            else:
                support_resistance = "Mid-range"
            
            # Volatility regime
            volatility = close.pct_change().std() * 100
            if volatility > 2.0:
                volatility_regime = "High volatility"
            elif volatility < 0.5:
                volatility_regime = "Low volatility"
            else:
                volatility_regime = "Normal volatility"
            
            return {
                'momentum_14d': momentum_14d,
                'volume_trend': volume_trend,
                'support_resistance': support_resistance,
                'volatility_regime': volatility_regime
            }
            
        except Exception as e:
            logger.warning(f"Error calculating technical sentiment: {e}")
            return {
                'momentum_14d': 0.0,
                'volume_trend': 'Unknown',
                'support_resistance': 'Unknown',
                'volatility_regime': 'Unknown'
            }
    
    def _fallback_analysis(self, symbol: str, available_strategies: List[str]) -> Dict[str, Any]:
        """Fallback analysis when GPT analysis fails"""
        return {
            'symbol': symbol,
            'timestamp': datetime.utcnow().isoformat(),
            'market_conditions': {'regime': 'UNKNOWN'},
            'recommended_strategies': available_strategies[:3],  # Return first 3 strategies
            'confidence_score': 0.3,
            'error': 'ChatGPT analysis unavailable, using fallback',
            'fallback_mode': True
        }

# Export the main class
__all__ = ['ChatGPTStrategyOptimizer']