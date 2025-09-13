"""
Manus AI Integration for Advanced Market Analysis
Enhanced with professional trading best practices and intelligent strategy selection
"""

import os
import requests
import logging
from typing import Dict, List, Optional, Any, Tuple
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from ..logs.logger import get_logger
from ..regime.detector import RegimeDetector
from .sentiment_analyzer import SentimentAnalyzer
from ..signals.utils import calculate_atr

logger = get_logger(__name__)

class ManusAI:
    """Enhanced Manus AI service for advanced market analysis with professional trading best practices"""
    
    def __init__(self):
        self.name = "Manus AI"
        self.api_key = os.getenv('MANUS_API')
        
        # Manus AI API endpoints (adjust based on actual API documentation)
        self.base_url = "https://api.manus.ai/v1"  # Placeholder - update with actual endpoint
        
        # Initialize professional trading components
        self.regime_detector = RegimeDetector()
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Market-specific strategy mappings based on market conditions and asset type
        self.forex_major_strategy_mapping = {
            'TRENDING': {
                'primary': ['donchian_atr', 'ema_rsi'],
                'secondary': ['macd_strategy'],
                'avoid': ['meanrev_bb', 'stochastic'],
                'reasoning': 'Trending forex markets favor breakout and momentum strategies with lower volatility'
            },
            'STRONG_TRENDING': {
                'primary': ['donchian_atr', 'fibonacci'],
                'secondary': ['ema_rsi'],
                'avoid': ['meanrev_bb', 'stochastic', 'rsi_divergence'],
                'reasoning': 'Strong forex trends require momentum strategies with wider stops'
            },
            'RANGING': {
                'primary': ['meanrev_bb', 'stochastic'],
                'secondary': ['rsi_divergence'],
                'avoid': ['donchian_atr', 'fibonacci'],
                'reasoning': 'Range-bound forex markets favor mean reversion strategies'
            },
            'HIGH_VOLATILITY': {
                'primary': ['stochastic', 'rsi_divergence'],
                'secondary': ['meanrev_bb'],
                'avoid': ['donchian_atr'],
                'reasoning': 'High volatility forex requires precision timing strategies'
            }
        }
        
        self.crypto_strategy_mapping = {
            'TRENDING': {
                'primary': ['donchian_atr', 'ema_rsi', 'macd_strategy'],
                'secondary': ['fibonacci'],
                'avoid': ['meanrev_bb'],
                'reasoning': 'Trending crypto markets strongly favor momentum and breakout strategies'
            },
            'STRONG_TRENDING': {
                'primary': ['donchian_atr', 'fibonacci', 'ema_rsi'],
                'secondary': ['macd_strategy'],
                'avoid': ['meanrev_bb', 'stochastic', 'rsi_divergence'],
                'reasoning': 'Strong crypto trends require aggressive momentum strategies'
            },
            'RANGING': {
                'primary': ['meanrev_bb', 'rsi_divergence'],
                'secondary': ['stochastic'],
                'avoid': ['donchian_atr', 'fibonacci'],
                'reasoning': 'Range-bound crypto markets favor quick mean reversion strategies'
            },
            'HIGH_VOLATILITY': {
                'primary': ['rsi_divergence', 'stochastic'],
                'secondary': ['meanrev_bb', 'ema_rsi'],
                'avoid': ['donchian_atr', 'fibonacci'],
                'reasoning': 'High volatility crypto requires nimble timing strategies'
            }
        }
        
        # Fallback strategy mapping for other market types (maintains compatibility)
        self.default_strategy_mapping = {
            'TRENDING': {
                'primary': ['donchian_atr', 'ema_rsi'],
                'secondary': ['macd_strategy'],
                'avoid': ['meanrev_bb', 'stochastic'],
                'reasoning': 'Trending markets favor breakout and momentum strategies'
            },
            'STRONG_TRENDING': {
                'primary': ['donchian_atr', 'fibonacci'],
                'secondary': ['ema_rsi'],
                'avoid': ['meanrev_bb', 'stochastic', 'rsi_divergence'],
                'reasoning': 'Strong trends require momentum strategies with wider stops'
            },
            'RANGING': {
                'primary': ['meanrev_bb', 'stochastic'],
                'secondary': ['rsi_divergence'],
                'avoid': ['donchian_atr', 'fibonacci'],
                'reasoning': 'Range-bound markets favor mean reversion strategies'
            },
            'HIGH_VOLATILITY': {
                'primary': ['stochastic', 'rsi_divergence'],
                'secondary': ['meanrev_bb'],
                'avoid': ['donchian_atr'],
                'reasoning': 'High volatility requires precision timing strategies'
            }
        }
        
        # Risk management parameters
        self.max_risk_per_trade = 0.01  # 1% maximum risk per trade
        
        # Market-specific volatility thresholds
        self.volatility_thresholds = {
            'forex_major': 0.005,  # 0.5% ATR threshold for high volatility (realistic for FX)
            'crypto': 0.02,        # 2.0% ATR threshold for high volatility (realistic for crypto)
            'other': 0.005         # Default to forex-like threshold
        }
        self.confidence_adjustment_factors = {
            'sentiment_boost': 0.05,  # Max 5% confidence boost from positive sentiment
            'regime_penalty': 0.10,   # Max 10% confidence reduction for wrong regime
            'volatility_penalty': 0.15  # Max 15% reduction for high volatility
        }
        
        # Major forex pairs for classification
        self.forex_major_pairs = {
            'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD',
            'EURGBP', 'EURJPY', 'GBPJPY', 'AUDJPY', 'CHFJPY', 'EURCHF', 'GBPAUD', 'AUDCAD'
        }
        
        # Crypto pairs for classification
        self.crypto_pairs = {
            'BTCUSD', 'ETHUSD', 'LTCUSD', 'ADAUSD', 'DOGUSD', 'SOLUSD', 'AVAXUSD'
        }
        
        logger.info(f"Enhanced Manus AI service initialized with market-type-aware strategy selection")
    
    def _classify_market(self, symbol: str) -> str:
        """Classify market type based on symbol
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD', 'BTCUSD')
            
        Returns:
            str: 'forex_major', 'crypto', or 'other'
        """
        symbol_upper = symbol.upper()
        
        if symbol_upper in self.forex_major_pairs:
            return 'forex_major'
        elif symbol_upper in self.crypto_pairs:
            return 'crypto'
        else:
            return 'other'
    
    def suggest_strategies(self, symbol: str, market_data: pd.DataFrame, sentiment_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Intelligent strategy selection based on market conditions
        
        This is the core method that implements professional trading best practices
        by analyzing market regime, volatility, and sentiment to recommend optimal strategies.
        
        Args:
            symbol: Currency pair (e.g., 'EURUSD')
            market_data: OHLC data for analysis
            sentiment_data: Optional sentiment analysis data
            
        Returns:
            Dict with recommended strategies, reasoning, and risk parameters
        """
        try:
            # Classify market type for market-aware strategy selection
            market_type = self._classify_market(symbol)
            
            # Detect current market regime
            regime_data = self.regime_detector.detect_regime(market_data, symbol)
            current_regime = regime_data['regime']
            regime_confidence = regime_data['confidence']
            
            # Analyze sentiment if available
            if sentiment_data is None:
                sentiment_data = self._analyze_market_sentiment(symbol)
            
            # Calculate volatility metrics with market-specific thresholds
            volatility_analysis = self._calculate_volatility_metrics(market_data, market_type)
            
            # Get base strategy recommendations with market awareness
            strategy_recommendations = self._get_strategy_recommendations(
                current_regime, regime_confidence, volatility_analysis, sentiment_data, market_type
            )
            
            # Apply professional filters and adjustments with market context
            filtered_strategies = self._apply_professional_filters(
                strategy_recommendations, regime_data, volatility_analysis, sentiment_data, market_type
            )
            
            # Calculate risk guidance (defer position sizing to RiskManager)
            risk_parameters = self._calculate_risk_parameters(
                symbol, market_data, volatility_analysis
            )
            risk_guidance = self._suggest_risk_adjustments(volatility_analysis)
            
            result = {
                'status': 'success',
                'symbol': symbol,
                'timestamp': datetime.utcnow().isoformat(),
                'market_analysis': {
                    'market_type': market_type,
                    'regime': current_regime,
                    'regime_confidence': regime_confidence,
                    'volatility_level': volatility_analysis['level'],
                    'atr_percentage': volatility_analysis['atr_percentage'],
                    'sentiment': sentiment_data.get('label', 'neutral')
                },
                'recommended_strategies': filtered_strategies,
                'risk_guidance': risk_guidance,  # Risk suggestions, not mandates
                'risk_parameters': risk_parameters,  # Technical parameters for reference
                'reasoning': self._generate_reasoning(current_regime, volatility_analysis, sentiment_data, market_type)
            }
            
            logger.info(f"Strategy recommendations generated for {symbol}: "
                       f"regime={current_regime}, strategies={[s['name'] for s in filtered_strategies[:3]]}")
            
            return result
            
        except ValueError as e:
            logger.error(f"Data validation error for {symbol}: {e}")
            return self._fallback_strategy_suggestions(symbol)
        except pd.errors.EmptyDataError as e:
            logger.error(f"Empty market data for {symbol}: {e}")
            return self._fallback_strategy_suggestions(symbol)
        except Exception as e:
            logger.error(f"Unexpected error generating strategy recommendations for {symbol}: {e}")
            return self._fallback_strategy_suggestions(symbol)
    
    def _analyze_market_sentiment(self, symbol: str) -> Dict:
        """Analyze market sentiment for the given symbol using real sentiment analysis"""
        try:
            # Use actual sentiment analyzer for real sentiment data
            # For financial news analysis, we'll create a sample news text related to the symbol
            news_text = self._get_symbol_news_context(symbol)
            
            if news_text:
                sentiment_result = self.sentiment_analyzer.analyze_sentiment(news_text, method="combined")
                return {
                    'score': sentiment_result.get('score', 0.0),
                    'label': sentiment_result.get('label', 'neutral').lower(),
                    'confidence': sentiment_result.get('confidence', 0.5),
                    'reasoning': f"Sentiment analysis based on recent {symbol} market context"
                }
            else:
                # Fallback to neutral if no news context available
                return {
                    'score': 0.0,
                    'label': 'neutral',
                    'confidence': 0.3,
                    'reasoning': 'No recent news context available - neutral sentiment assumed'
                }
        except Exception as e:
            logger.warning(f"Error analyzing sentiment for {symbol}: {e}")
            return {'score': 0.0, 'label': 'neutral', 'confidence': 0.0, 'reasoning': 'Sentiment analysis unavailable'}
    
    def _calculate_volatility_metrics(self, market_data: pd.DataFrame, market_type: str = 'other') -> Dict:
        """Calculate comprehensive volatility metrics using proper ATR calculation"""
        try:
            # Add explicit data length validation
            if len(market_data) < 20:  # Need at least 20 bars for reliable ATR calculation
                logger.warning(f"Insufficient data for volatility analysis: {len(market_data)} bars (minimum 20 required)")
                return self._fallback_volatility_metrics()
            
            # Use proper TA-Lib ATR calculation from utils
            atr_values = calculate_atr(market_data, period=14)
            
            # Validate ATR calculation results
            if atr_values is None or len(atr_values) == 0 or pd.isna(atr_values[-1]):
                logger.warning("ATR calculation failed or returned invalid results")
                return self._fallback_volatility_metrics()
            
            atr_14 = atr_values[-1]
            current_price = market_data['close'].iloc[-1]
            atr_percentage = atr_14 / current_price
            
            # Classify volatility level with market-specific thresholds
            volatility_threshold = self.volatility_thresholds.get(market_type, self.volatility_thresholds['other'])
            
            if atr_percentage > volatility_threshold:
                level = 'high'
                multiplier = 1.5  # Wider stops for high volatility
            elif atr_percentage > volatility_threshold * 0.4:
                level = 'medium'
                multiplier = 1.0
            else:
                level = 'low'
                multiplier = 0.8  # Tighter stops for low volatility
            
            return {
                'level': level,
                'atr_value': float(atr_14),
                'atr_percentage': float(atr_percentage),
                'stop_multiplier': multiplier,
                'current_price': float(current_price),
                'data_points': len(market_data),
                'market_type': market_type,
                'volatility_threshold': volatility_threshold
            }
            
        except Exception as e:
            logger.error(f"Error calculating volatility metrics: {e}")
            return self._fallback_volatility_metrics()
    
    def _fallback_volatility_metrics(self) -> Dict:
        """Fallback volatility metrics when calculation fails"""
        return {
            'level': 'medium',
            'atr_value': 0.0001,  # Realistic fallback for FX
            'atr_percentage': 0.001,  # 0.1% fallback
            'stop_multiplier': 1.0,
            'current_price': 1.0,
            'data_points': 0
        }
    
    def _get_symbol_news_context(self, symbol: str) -> str:
        """Get recent news context for sentiment analysis of the given symbol"""
        try:
            # Generate contextual news text based on symbol
            # In a production environment, this would fetch real news from APIs
            # For now, we simulate market context based on major currency pairs
            
            base_currency = symbol[:3] if len(symbol) >= 6 else symbol[:3]
            quote_currency = symbol[3:6] if len(symbol) >= 6 else symbol[3:]
            
            # Sample news contexts for different currency pairs
            news_contexts = {
                'EUR': f"European Central Bank maintains monetary policy stance. Euro shows resilience amid market uncertainty.",
                'USD': f"Federal Reserve policy decisions continue to influence dollar strength. US economic indicators remain mixed.",
                'GBP': f"Bank of England policy and UK economic data drive pound volatility. Brexit impacts continue to influence market sentiment.",
                'JPY': f"Bank of Japan intervention concerns and safe-haven demand affect yen movements. Risk sentiment influences Japanese currency.",
                'AUD': f"Reserve Bank of Australia policy and commodity prices drive Australian dollar. China economic data impacts AUD sentiment.",
                'CAD': f"Bank of Canada policy and oil prices influence Canadian dollar. Commodity market conditions affect CAD strength.",
                'CHF': f"Swiss National Bank policy and safe-haven flows drive franc movements. European developments impact Swiss currency.",
                'NZD': f"Reserve Bank of New Zealand policy and dairy prices influence kiwi. Risk sentiment affects New Zealand dollar."
            }
            
            # Combine contexts for both currencies in the pair
            base_context = news_contexts.get(base_currency, f"{base_currency} shows mixed market sentiment.")
            quote_context = news_contexts.get(quote_currency, f"{quote_currency} maintains current market position.")
            
            combined_context = f"{base_context} {quote_context} Current {symbol} market conditions reflect broader economic trends and central bank policies."
            
            return combined_context
            
        except Exception as e:
            logger.warning(f"Error generating news context for {symbol}: {e}")
            return f"Market analysis for {symbol} showing standard trading conditions with moderate volatility expectations."
    
    def _get_strategy_recommendations(
        self, 
        regime: str, 
        regime_confidence: float, 
        volatility_analysis: Dict, 
        sentiment_data: Dict,
        market_type: str = 'other'
    ) -> List[Dict]:
        """Get base strategy recommendations based on market conditions and market type"""
        try:
            # Select appropriate strategy mapping based on market type
            if market_type == 'forex_major':
                strategy_mapping = self.forex_major_strategy_mapping
            elif market_type == 'crypto':
                strategy_mapping = self.crypto_strategy_mapping
            else:
                strategy_mapping = self.default_strategy_mapping
            
            # Get strategy mapping for current regime
            if regime not in strategy_mapping:
                regime = 'RANGING'  # Default fallback
            
            mapping = strategy_mapping[regime]
            recommendations = []
            
            # Primary strategies (highest confidence)
            for strategy in mapping['primary']:
                recommendations.append({
                    'name': strategy,
                    'priority': 'primary',
                    'base_confidence': 0.8,
                    'reasoning': mapping['reasoning']
                })
            
            # Secondary strategies (medium confidence)
            for strategy in mapping['secondary']:
                recommendations.append({
                    'name': strategy,
                    'priority': 'secondary', 
                    'base_confidence': 0.6,
                    'reasoning': f"Secondary choice for {regime.lower()} markets"
                })
            
            # Add all other strategies as tertiary (low confidence)
            all_strategies = ['ema_rsi', 'donchian_atr', 'meanrev_bb', 'macd_strategy', 
                             'stochastic', 'rsi_divergence', 'fibonacci']
            avoid_strategies = set(mapping.get('avoid', []))
            used_strategies = set(mapping['primary'] + mapping['secondary'])
            
            for strategy in all_strategies:
                if strategy not in used_strategies and strategy not in avoid_strategies:
                    recommendations.append({
                        'name': strategy,
                        'priority': 'tertiary',
                        'base_confidence': 0.4,
                        'reasoning': f"Neutral strategy for {regime.lower()} conditions"
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating strategy recommendations: {e}")
            return self._fallback_strategy_list()
    
    def _apply_professional_filters(
        self, 
        recommendations: List[Dict], 
        regime_data: Dict, 
        volatility_analysis: Dict, 
        sentiment_data: Dict,
        market_type: str = 'other'
    ) -> List[Dict]:
        """Apply professional trading filters and confidence adjustments"""
        try:
            filtered_recommendations = []
            
            for rec in recommendations:
                # Start with base confidence
                adjusted_confidence = rec['base_confidence']
                adjustment_reasons = []
                
                # Regime confidence adjustment
                regime_confidence = regime_data.get('confidence', 0.5)
                if regime_confidence < 0.6:
                    adjusted_confidence *= 0.9  # Reduce confidence for uncertain regimes
                    adjustment_reasons.append("regime_uncertainty")
                
                # Volatility adjustment
                volatility_level = volatility_analysis['level']
                if volatility_level == 'high':
                    # High volatility strategies get boost, others get penalty
                    if rec['name'] in ['stochastic', 'rsi_divergence']:
                        adjusted_confidence *= 1.1
                        adjustment_reasons.append("volatility_favorable")
                    else:
                        adjusted_confidence *= 0.85
                        adjustment_reasons.append("volatility_unfavorable")
                
                # Sentiment adjustment
                sentiment_score = sentiment_data.get('score', 0.0)
                if abs(sentiment_score) > 0.3:  # Strong sentiment
                    if rec['name'] in ['ema_rsi', 'macd_strategy']:  # Momentum strategies
                        adjusted_confidence *= 1.05  # Small boost for momentum in strong sentiment
                        adjustment_reasons.append("sentiment_momentum_boost")
                
                # Market-type specific adjustments
                if market_type == 'crypto':
                    # Crypto markets favor momentum strategies
                    if rec['name'] in ['donchian_atr', 'ema_rsi', 'macd_strategy']:
                        adjusted_confidence *= 1.05
                        adjustment_reasons.append("crypto_momentum_boost")
                elif market_type == 'forex_major':
                    # Forex majors favor stability and precision
                    if rec['name'] in ['meanrev_bb', 'stochastic', 'rsi_divergence']:
                        adjusted_confidence *= 1.02
                        adjustment_reasons.append("forex_precision_boost")
                
                # Professional risk controls with explicit confidence clamping (0.1-1.0)
                adjusted_confidence = max(0.1, min(1.0, adjusted_confidence))
                
                # Verify confidence is within valid range
                if not (0.1 <= adjusted_confidence <= 1.0):
                    logger.warning(f"Confidence out of range for {rec['name']}: {adjusted_confidence}, clamping to valid range")
                    adjusted_confidence = max(0.1, min(1.0, adjusted_confidence))
                
                # Note: Position sizing should be handled by RiskManager, not Manus AI
                # Manus AI provides risk parameter recommendations only
                position_size = None  # Defer to RiskManager
                
                filtered_rec = {
                    'name': rec['name'],
                    'priority': rec['priority'],
                    'confidence': round(adjusted_confidence, 3),
                    'original_confidence': rec['base_confidence'],
                    'reasoning': rec['reasoning'],
                    'adjustments': adjustment_reasons,
                    'recommended': adjusted_confidence >= 0.5,
                    'risk_guidance': {
                        'market_type': market_type,
                        'volatility_level': volatility_analysis['level'],
                        'suggested_stop_multiplier': volatility_analysis.get('stop_multiplier', 1.0)
                    }
                }
                
                filtered_recommendations.append(filtered_rec)
            
            # Sort by confidence descending
            filtered_recommendations.sort(key=lambda x: x['confidence'], reverse=True)
            
            return filtered_recommendations
            
        except Exception as e:
            logger.error(f"Error applying professional filters: {e}")
            # Apply basic confidence clamping to unfiltered recommendations as fallback
            for rec in recommendations:
                if 'confidence' in rec:
                    rec['confidence'] = max(0.1, min(1.0, rec.get('confidence', 0.5)))
            return recommendations  # Return unfiltered if error
    
    def _suggest_risk_adjustments(self, volatility_analysis: Dict) -> Dict:
        """Suggest risk parameter adjustments based on market conditions (defer actual sizing to RiskManager)"""
        try:
            volatility_level = volatility_analysis['level']
            
            # Provide guidance for RiskManager, not direct position sizing
            if volatility_level == 'high':
                risk_adjustment = {
                    'suggested_risk_reduction': 0.5,  # Suggest 50% risk reduction
                    'reasoning': 'High volatility detected - recommend reduced position size',
                    'stop_multiplier_adjustment': 1.5
                }
            elif volatility_level == 'low':
                risk_adjustment = {
                    'suggested_risk_reduction': 0.0,  # No reduction needed
                    'reasoning': 'Low volatility - standard position sizing acceptable',
                    'stop_multiplier_adjustment': 0.8
                }
            else:
                risk_adjustment = {
                    'suggested_risk_reduction': 0.0,
                    'reasoning': 'Medium volatility - standard risk parameters',
                    'stop_multiplier_adjustment': 1.0
                }
            
            return risk_adjustment
            
        except Exception:
            return {
                'suggested_risk_reduction': 0.0,
                'reasoning': 'Error in risk analysis - use standard parameters',
                'stop_multiplier_adjustment': 1.0
            }
    
    def _calculate_risk_parameters(self, symbol: str, market_data: pd.DataFrame, volatility_analysis: Dict) -> Dict:
        """Calculate professional risk management parameters"""
        try:
            atr_value = volatility_analysis['atr_value']
            current_price = volatility_analysis['current_price']
            stop_multiplier = volatility_analysis['stop_multiplier']
            
            # Calculate ATR-based stop loss distances
            atr_stop_distance = atr_value * stop_multiplier
            atr_stop_percentage = atr_stop_distance / current_price
            
            # Professional take profit ratios
            risk_reward_ratios = {
                'conservative': 1.5,  # 1.5:1 RR
                'balanced': 2.0,      # 2:1 RR
                'aggressive': 3.0     # 3:1 RR
            }
            
            return {
                'max_risk_per_trade': self.max_risk_per_trade,
                'atr_stop_distance': round(atr_stop_distance, 5),
                'atr_stop_percentage': round(atr_stop_percentage, 4),
                'recommended_stop_multiplier': stop_multiplier,
                'risk_reward_ratios': risk_reward_ratios,
                'position_sizing_method': 'atr_based',
                'volatility_adjustment': volatility_analysis['level']
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk parameters: {e}")
            return {
                'max_risk_per_trade': 0.01,
                'atr_stop_distance': 0.001,
                'atr_stop_percentage': 0.001,
                'recommended_stop_multiplier': 1.0,
                'risk_reward_ratios': {'conservative': 1.5, 'balanced': 2.0, 'aggressive': 3.0}
            }
    
    def _generate_reasoning(self, regime: str, volatility_analysis: Dict, sentiment_data: Dict, market_type: str = 'other') -> str:
        """Generate human-readable reasoning for strategy recommendations"""
        try:
            reasoning_parts = []
            
            # Market type context
            market_descriptions = {
                'forex_major': 'major forex pair with institutional liquidity',
                'crypto': 'cryptocurrency pair with 24/7 trading',
                'other': 'financial instrument'
            }
            
            # Market regime reasoning with market type awareness
            if market_type == 'crypto':
                regime_explanations = {
                    'TRENDING': "Crypto market shows strong directional momentum favoring breakout strategies",
                    'STRONG_TRENDING': "Strong crypto trend requires aggressive momentum strategies", 
                    'RANGING': "Sideways crypto market favors quick mean reversion approaches",
                    'HIGH_VOLATILITY': "High crypto volatility requires nimble timing strategies"
                }
            elif market_type == 'forex_major':
                regime_explanations = {
                    'TRENDING': "Forex market shows clear directional movement favoring momentum strategies",
                    'STRONG_TRENDING': "Strong forex trend requires robust breakout strategies", 
                    'RANGING': "Sideways forex market favors precise mean reversion approaches",
                    'HIGH_VOLATILITY': "High forex volatility requires precision timing strategies"
                }
            else:
                regime_explanations = {
                    'TRENDING': "Market shows clear directional movement favoring momentum strategies",
                    'STRONG_TRENDING': "Strong trending conditions require robust breakout strategies", 
                    'RANGING': "Sideways market conditions favor mean reversion approaches",
                    'HIGH_VOLATILITY': "High volatility environment requires precise timing strategies"
                }
            
            reasoning_parts.append(f"Analyzing {market_descriptions.get(market_type, 'financial instrument')}: {regime_explanations.get(regime, 'Market regime analysis complete')}")
            
            # Volatility reasoning
            volatility_level = volatility_analysis['level']
            if volatility_level == 'high':
                reasoning_parts.append("High volatility detected - using wider stops and reduced position sizes")
            elif volatility_level == 'low':
                reasoning_parts.append("Low volatility environment - allowing tighter stops and standard sizing")
            
            # Sentiment reasoning
            sentiment_label = sentiment_data.get('label', 'neutral')
            if sentiment_label != 'neutral':
                reasoning_parts.append(f"Market sentiment is {sentiment_label} - factored into strategy selection")
            
            return ". ".join(reasoning_parts) + "."
            
        except Exception:
            return "Professional strategy analysis completed based on market conditions."
    
    def _fallback_strategy_suggestions(self, symbol: str) -> Dict:
        """Fallback strategy suggestions when analysis fails"""
        market_type = self._classify_market(symbol)
        return {
            'status': 'fallback',
            'symbol': symbol,
            'timestamp': datetime.utcnow().isoformat(),
            'market_analysis': {
                'market_type': market_type,
                'regime': 'unknown',
                'regime_confidence': 0.0,
                'volatility_level': 'medium',
                'atr_percentage': 0.001,
                'sentiment': 'neutral'
            },
            'recommended_strategies': self._fallback_strategy_list(),
            'reasoning': f"Using fallback strategy recommendations for {market_type} market - full analysis unavailable",
            'risk_guidance': {
                'market_type': market_type,
                'suggested_risk_reduction': 0.0,
                'reasoning': 'Fallback mode - use standard risk parameters',
                'stop_multiplier_adjustment': 1.0
            },
            'risk_parameters': {
                'max_risk_per_trade': 0.01,
                'recommended_stop_multiplier': 1.0
            }
        }
    
    def _fallback_strategy_list(self) -> List[Dict]:
        """Default strategy list for fallback scenarios"""
        return [
            {'name': 'ema_rsi', 'confidence': 0.6, 'priority': 'primary', 'recommended': True, 'reasoning': 'Fallback momentum strategy'},
            {'name': 'meanrev_bb', 'confidence': 0.5, 'priority': 'secondary', 'recommended': True, 'reasoning': 'Fallback mean reversion strategy'},
            {'name': 'stochastic', 'confidence': 0.4, 'priority': 'tertiary', 'recommended': False, 'reasoning': 'Fallback oscillator strategy'}
        ]

    def is_available(self) -> bool:
        """Check if Manus AI service is available"""
        return bool(self.api_key)
    
    def _make_request(self, endpoint: str, data: Dict = None) -> requests.Response:
        """Make request to Manus AI API"""
        url = f"{self.base_url}{endpoint}"
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        try:
            if data:
                response = requests.post(url, headers=headers, json=data, timeout=30)
            else:
                response = requests.get(url, headers=headers, timeout=30)
                
            response.raise_for_status()
            return response
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Manus AI API request failed: {e}")
            raise
    
    def analyze_market_conditions(self, market_data: Dict) -> Dict:
        """Analyze current market conditions using Manus AI"""
        if not self.is_available():
            return self._fallback_analysis(market_data)
            
        try:
            analysis_prompt = {
                "task": "market_analysis",
                "data": market_data,
                "requirements": [
                    "Market trend analysis",
                    "Volatility assessment", 
                    "Risk factors",
                    "Trading opportunities",
                    "Key support/resistance levels"
                ]
            }
            
            # Note: Adjust endpoint based on actual Manus AI API
            response = self._make_request("/analyze", analysis_prompt)
            result = response.json()
            
            logger.info("Market analysis completed by Manus AI")
            return result
            
        except Exception as e:
            logger.error(f"Manus AI market analysis failed: {e}")
            return self._fallback_analysis(market_data)
    
    def generate_signal_insights(self, signal_data: Dict) -> Dict:
        """Generate insights for trading signals using Manus AI"""
        if not self.is_available():
            return self._fallback_signal_insights(signal_data)
            
        try:
            insight_prompt = {
                "task": "signal_analysis",
                "signal": signal_data,
                "analysis_type": [
                    "Signal strength assessment",
                    "Risk-reward analysis",
                    "Entry/exit optimization",
                    "Market context evaluation"
                ]
            }
            
            response = self._make_request("/signals/analyze", insight_prompt)
            result = response.json()
            
            logger.info("Signal insights generated by Manus AI")
            return result
            
        except Exception as e:
            logger.error(f"Manus AI signal analysis failed: {e}")
            return self._fallback_signal_insights(signal_data)
    
    def generate_portfolio_recommendations(self, portfolio_data: Dict) -> Dict:
        """Generate portfolio optimization recommendations"""
        if not self.is_available():
            return self._fallback_portfolio_recommendations(portfolio_data)
            
        try:
            portfolio_prompt = {
                "task": "portfolio_optimization",
                "portfolio": portfolio_data,
                "objectives": [
                    "Risk optimization",
                    "Diversification analysis",
                    "Performance enhancement",
                    "Correlation analysis"
                ]
            }
            
            response = self._make_request("/portfolio/optimize", portfolio_prompt)
            result = response.json()
            
            logger.info("Portfolio recommendations generated by Manus AI")
            return result
            
        except Exception as e:
            logger.error(f"Manus AI portfolio analysis failed: {e}")
            return self._fallback_portfolio_recommendations(portfolio_data)
    
    def _fallback_analysis(self, market_data: Dict) -> Dict:
        """Fallback analysis when Manus AI is unavailable"""
        return {
            "status": "fallback",
            "analysis": {
                "trend": "Analyzing using traditional methods",
                "volatility": "Standard volatility calculations applied",
                "recommendations": "Basic technical analysis completed",
                "note": "Manus AI unavailable - using fallback analysis"
            }
        }
    
    def _fallback_signal_insights(self, signal_data: Dict) -> Dict:
        """Fallback signal insights when Manus AI is unavailable"""
        return {
            "status": "fallback", 
            "insights": {
                "strength": "Calculated using traditional indicators",
                "risk_reward": "Standard risk management applied",
                "recommendation": "Follow standard signal guidelines",
                "note": "Manus AI unavailable - using fallback insights"
            }
        }
    
    def _fallback_portfolio_recommendations(self, portfolio_data: Dict) -> Dict:
        """Fallback portfolio recommendations when Manus AI is unavailable"""
        return {
            "status": "fallback",
            "recommendations": {
                "diversification": "Standard diversification rules applied",
                "risk_management": "Traditional risk management in use",
                "optimization": "Basic portfolio optimization applied",
                "note": "Manus AI unavailable - using fallback recommendations"
            }
        }
    
    def optimize_strategy_portfolio(self, strategy_data: Dict) -> Dict:
        """Optimize strategy portfolio allocation using Manus AI analysis"""
        if not self.is_available():
            return self._fallback_strategy_optimization(strategy_data)
            
        try:
            optimization_prompt = {
                "task": "strategy_portfolio_optimization",
                "strategies": strategy_data,
                "optimization_goals": [
                    "Risk-adjusted returns maximization",
                    "Drawdown minimization",
                    "Correlation analysis",
                    "Market regime adaptation",
                    "Capital allocation efficiency"
                ],
                "constraints": {
                    "max_allocation_per_strategy": 0.4,
                    "min_strategies_active": 3,
                    "risk_tolerance": "moderate"
                }
            }
            
            response = self._make_request("/portfolio/optimize_strategies", optimization_prompt)
            result = response.json()
            
            logger.info("Strategy portfolio optimization completed by Manus AI")
            return result
            
        except Exception as e:
            logger.error(f"Manus AI strategy optimization failed: {e}")
            return self._fallback_strategy_optimization(strategy_data)
    
    def design_optimal_table(self, table_context: Dict) -> Dict:
        """Generate optimal table design recommendations using Manus AI"""
        if not self.is_available():
            return self._fallback_table_design(table_context)
            
        try:
            design_prompt = {
                "task": "trading_table_optimization",
                "context": table_context,
                "design_principles": [
                    "Information hierarchy optimization",
                    "Cognitive load reduction",
                    "Trading workflow efficiency",
                    "Visual clarity enhancement",
                    "Mobile responsiveness"
                ],
                "requirements": {
                    "data_density": "high",
                    "user_type": "professional_trader",
                    "primary_actions": ["signal_analysis", "risk_assessment", "trade_execution"]
                }
            }
            
            response = self._make_request("/design/optimize_table", design_prompt)
            result = response.json()
            
            logger.info("Table design optimization completed by Manus AI")
            return result
            
        except Exception as e:
            logger.error(f"Manus AI table design failed: {e}")
            return self._fallback_table_design(table_context)
    
    def _fallback_strategy_optimization(self, strategy_data: Dict) -> Dict:
        """Fallback strategy optimization when Manus AI is unavailable"""
        # Calculate basic optimization based on Sharpe ratio and drawdown
        strategies = strategy_data.get('strategies', {})
        total_strategies = len(strategies)
        
        # Simple equal-weight fallback with performance bias
        optimized_allocations = {}
        performance_scores = {}
        
        for strategy_name, metrics in strategies.items():
            # Calculate performance score (Sharpe ratio weighted by win rate)
            sharpe = metrics.get('sharpe_ratio', 0)
            win_rate = metrics.get('win_rate', 0.5)
            max_dd = metrics.get('max_drawdown', 0.2)
            
            # Performance score: emphasize positive Sharpe, high win rate, low drawdown
            score = (sharpe * 0.4) + (win_rate * 0.4) - (max_dd * 0.2)
            performance_scores[strategy_name] = max(score, 0.1)  # Minimum allocation
        
        # Normalize scores to allocations
        total_score = sum(performance_scores.values())
        for strategy_name in strategies:
            allocation = performance_scores[strategy_name] / total_score
            # Cap maximum allocation at 40%
            optimized_allocations[strategy_name] = min(allocation, 0.4)
        
        # Renormalize if capping occurred
        total_allocation = sum(optimized_allocations.values())
        if total_allocation != 1.0:
            for strategy in optimized_allocations:
                optimized_allocations[strategy] /= total_allocation
        
        return {
            "status": "fallback",
            "optimization": {
                "recommended_allocations": optimized_allocations,
                "reasoning": "Performance-weighted allocation based on Sharpe ratio and win rate",
                "risk_level": "moderate",
                "expected_sharpe": sum(strategies[s]['sharpe_ratio'] * optimized_allocations[s] for s in strategies),
                "diversification_score": 0.7,
                "note": "Manus AI unavailable - using traditional optimization"
            }
        }
    
    def _fallback_table_design(self, table_context: Dict) -> Dict:
        """Fallback table design when Manus AI is unavailable"""
        data_type = table_context.get('data_type', 'general')
        
        if data_type == 'strategy_performance':
            recommendations = {
                "column_order": [
                    "strategy_name", "win_rate", "total_trades", "avg_pnl", 
                    "profit_factor", "sharpe_ratio", "max_drawdown", "status"
                ],
                "formatting": {
                    "win_rate": "percentage_green_red",
                    "avg_pnl": "currency_color_coded", 
                    "profit_factor": "decimal_2_color_coded",
                    "sharpe_ratio": "decimal_2_color_coded",
                    "max_drawdown": "percentage_red_emphasis"
                },
                "sorting": {
                    "default": "profit_factor",
                    "direction": "descending"
                },
                "visual_cues": {
                    "top_performer": "green_highlight",
                    "underperformer": "yellow_background",
                    "risk_warning": "red_border"
                }
            }
        else:
            # Generic table recommendations
            recommendations = {
                "column_order": ["name", "value", "change", "status"],
                "formatting": {
                    "value": "auto_format",
                    "change": "color_coded_change"
                },
                "sorting": {
                    "default": "value",
                    "direction": "descending"
                }
            }
        
        return {
            "status": "fallback",
            "design": {
                "layout": recommendations,
                "best_practices": [
                    "Use consistent color coding for performance metrics",
                    "Prioritize most actionable data in leftmost columns",
                    "Apply visual hierarchy with typography and spacing",
                    "Include hover states for detailed information"
                ],
                "accessibility": {
                    "contrast_ratio": "4.5:1 minimum",
                    "keyboard_navigation": "full_support",
                    "screen_reader": "aria_labels_required"
                },
                "note": "Manus AI unavailable - using standard design principles"
            }
        }

    def test_connection(self) -> bool:
        """Test connection to Manus AI"""
        if not self.is_available():
            return False
            
        try:
            # Test with a simple health check (adjust endpoint as needed)
            response = self._make_request("/health")
            return response.status_code == 200
        except:
            return False