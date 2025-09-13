"""
AI Strategy Consensus System
Advanced consensus mechanism combining Manus AI and ChatGPT recommendations
with intelligent conflict resolution and weighted scoring
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from ..logs.logger import get_logger
from ..ai_capabilities import get_ai_capabilities, OPENAI_ENABLED
from .manus_ai import ManusAI

logger = get_logger(__name__)

# Check AI capabilities and conditionally import ChatGPT components
ai_capabilities = get_ai_capabilities()
CHATGPT_AVAILABLE = ai_capabilities['openai_enabled']

# Conditional imports
ChatGPTStrategyOptimizer = None

if CHATGPT_AVAILABLE:
    try:
        from .chatgpt_strategy_optimizer import ChatGPTStrategyOptimizer
        logger.info("AI Consensus: ChatGPT components loaded for dual-AI mode")
    except ImportError as e:
        logger.warning(f"AI Consensus: ChatGPT components failed to load: {e}")
        CHATGPT_AVAILABLE = False
else:
    logger.info("AI Consensus: Operating in fallback mode (Manus AI only)")

class ConsensusLevel(Enum):
    """Consensus agreement levels"""
    HIGH_AGREEMENT = "high_agreement"      # >80% alignment
    MODERATE_AGREEMENT = "moderate_agreement"  # 60-80% alignment
    LOW_AGREEMENT = "low_agreement"        # 40-60% alignment
    DISAGREEMENT = "disagreement"          # <40% alignment

@dataclass
class AIRecommendation:
    """Structured AI recommendation"""
    ai_name: str
    strategies: List[Dict[str, Any]]
    confidence: float
    reasoning: str
    timestamp: datetime
    market_conditions: Dict[str, Any]
    risk_assessment: Dict[str, Any]

@dataclass
class ConsensusResult:
    """Final consensus result"""
    recommended_strategies: List[Dict[str, Any]]
    consensus_level: ConsensusLevel
    overall_confidence: float
    agreement_score: float
    conflict_areas: List[str]
    resolution_method: str
    ai_contributions: Dict[str, Dict[str, Any]]
    reasoning: str
    timestamp: datetime

class AIStrategyConsensus:
    """
    Advanced AI Strategy Consensus System
    Combines Manus AI and ChatGPT recommendations using sophisticated consensus algorithms
    """
    
    def __init__(self):
        self.name = "AI Strategy Consensus System"
        
        # Initialize AI services (Manus AI always available)
        self.manus_ai = ManusAI()
        
        # Conditionally initialize ChatGPT components
        if CHATGPT_AVAILABLE and ChatGPTStrategyOptimizer:
            try:
                self.chatgpt_optimizer = ChatGPTStrategyOptimizer()
                logger.info("AIStrategyConsensus: Dual-AI mode active (Manus AI + ChatGPT)")
            except Exception as e:
                logger.warning(f"AIStrategyConsensus: Failed to initialize ChatGPT optimizer: {e}")
                self.chatgpt_optimizer = None
        else:
            self.chatgpt_optimizer = None
            logger.info("AIStrategyConsensus: Single-AI mode active (Manus AI only)")
        
        # Consensus configuration
        self.ai_weights = {
            'manus_ai': 0.5,  # Equal weight by default
            'chatgpt': 0.5
        }
        
        # Dynamic weight adjustment factors
        self.weight_adjustment_factors = {
            'historical_performance': 0.2,
            'confidence_levels': 0.3,
            'market_regime_accuracy': 0.3,
            'agreement_history': 0.2
        }
        
        # Consensus thresholds
        self.consensus_thresholds = {
            'high_agreement': 0.8,
            'moderate_agreement': 0.6,
            'low_agreement': 0.4
        }
        
        # Conflict resolution strategies
        self.conflict_resolution_methods = [
            'weighted_average',
            'best_confidence',
            'conservative_approach',
            'market_regime_based',
            'hybrid_ensemble'
        ]
        
        # Performance tracking
        self.ai_performance_history = {
            'manus_ai': {'accuracy': 0.7, 'reliability': 0.8, 'recent_performance': []},
            'chatgpt': {'accuracy': 0.75, 'reliability': 0.85, 'recent_performance': []}
        }
        
        logger.info("AI Strategy Consensus System initialized with dual-AI integration")
    
    async def generate_consensus_recommendation(
        self, 
        symbol: str, 
        market_data: pd.DataFrame,
        available_strategies: List[str],
        news_data: Optional[str] = None
    ) -> ConsensusResult:
        """
        Generate consensus recommendation combining both AI systems
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD', 'BTCUSD')
            market_data: OHLC price data
            available_strategies: List of available strategy names
            news_data: Optional news data for sentiment analysis
            
        Returns:
            ConsensusResult with final recommendations and analysis
        """
        try:
            logger.info(f"Generating AI consensus recommendation for {symbol}")
            
            # Get recommendations from both AI systems
            manus_recommendation = await self._get_manus_ai_recommendation(
                symbol, market_data, available_strategies
            )
            
            chatgpt_recommendation = await self._get_chatgpt_recommendation(
                symbol, market_data, available_strategies, news_data
            )
            
            # Calculate dynamic AI weights based on recent performance
            dynamic_weights = self._calculate_dynamic_weights(symbol, market_data)
            
            # Analyze agreement between AI systems
            agreement_analysis = self._analyze_ai_agreement(
                manus_recommendation, chatgpt_recommendation
            )
            
            # Determine consensus level
            consensus_level = self._determine_consensus_level(agreement_analysis['agreement_score'])
            
            # Apply conflict resolution if needed
            if consensus_level in [ConsensusLevel.LOW_AGREEMENT, ConsensusLevel.DISAGREEMENT]:
                resolution_result = self._resolve_conflicts(
                    manus_recommendation, chatgpt_recommendation, dynamic_weights, symbol
                )
            else:
                resolution_result = self._merge_agreements(
                    manus_recommendation, chatgpt_recommendation, dynamic_weights
                )
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(
                manus_recommendation, chatgpt_recommendation, agreement_analysis, dynamic_weights
            )
            
            # Generate final reasoning
            reasoning = self._generate_consensus_reasoning(
                manus_recommendation, chatgpt_recommendation, agreement_analysis, 
                consensus_level, resolution_result
            )
            
            # Create consensus result
            consensus_result = ConsensusResult(
                recommended_strategies=resolution_result['strategies'],
                consensus_level=consensus_level,
                overall_confidence=overall_confidence,
                agreement_score=agreement_analysis['agreement_score'],
                conflict_areas=agreement_analysis['conflict_areas'],
                resolution_method=resolution_result['method'],
                ai_contributions={
                    'manus_ai': {
                        'weight': dynamic_weights['manus_ai'],
                        'confidence': manus_recommendation.confidence,
                        'strategies': manus_recommendation.strategies[:3]
                    },
                    'chatgpt': {
                        'weight': dynamic_weights['chatgpt'],
                        'confidence': chatgpt_recommendation.confidence,
                        'strategies': chatgpt_recommendation.strategies[:3]
                    }
                },
                reasoning=reasoning,
                timestamp=datetime.utcnow()
            )
            
            # Update performance tracking
            self._update_performance_tracking(symbol, consensus_result)
            
            logger.info(f"AI consensus generated for {symbol}: {consensus_level.value} "
                       f"(confidence: {overall_confidence:.2f})")
            
            return consensus_result
            
        except Exception as e:
            logger.error(f"Error generating AI consensus for {symbol}: {e}")
            return self._fallback_consensus(symbol, available_strategies)
    
    async def validate_strategy_recommendation(
        self,
        symbol: str,
        strategy_name: str,
        market_data: pd.DataFrame,
        strategy_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Cross-validate a specific strategy recommendation using both AI systems
        
        Args:
            symbol: Trading symbol
            strategy_name: Name of strategy to validate
            market_data: Price data
            strategy_params: Strategy parameters
            
        Returns:
            Dict with validation results from both AIs
        """
        try:
            logger.info(f"Cross-validating strategy {strategy_name} for {symbol}")
            
            # Get Manus AI validation
            manus_validation = await self._validate_with_manus_ai(
                symbol, strategy_name, market_data, strategy_params
            )
            
            # Get ChatGPT validation
            chatgpt_validation = await self._validate_with_chatgpt(
                symbol, strategy_name, market_data, strategy_params
            )
            
            # Compare validations
            validation_agreement = self._compare_validations(manus_validation, chatgpt_validation)
            
            # Generate consensus validation
            consensus_validation = self._create_consensus_validation(
                manus_validation, chatgpt_validation, validation_agreement
            )
            
            result = {
                'symbol': symbol,
                'strategy_name': strategy_name,
                'timestamp': datetime.utcnow().isoformat(),
                'manus_validation': manus_validation,
                'chatgpt_validation': chatgpt_validation,
                'consensus_validation': consensus_validation,
                'validation_agreement': validation_agreement,
                'recommendation': consensus_validation['recommendation'],
                'confidence': consensus_validation['confidence']
            }
            
            logger.info(f"Strategy validation completed for {symbol} {strategy_name}: "
                       f"{consensus_validation['recommendation']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in strategy validation for {symbol} {strategy_name}: {e}")
            return {
                'error': str(e),
                'recommendation': 'inconclusive',
                'confidence': 0.0
            }
    
    async def optimize_strategy_portfolio(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        current_strategies: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Optimize strategy portfolio using dual-AI collaboration
        
        Args:
            symbol: Trading symbol
            market_data: Price data
            current_strategies: Current strategy configurations
            
        Returns:
            Dict with optimized portfolio recommendations
        """
        try:
            logger.info(f"Optimizing strategy portfolio for {symbol}")
            
            # Get Manus AI portfolio recommendations
            manus_portfolio = await self._get_manus_portfolio_optimization(
                symbol, market_data, current_strategies
            )
            
            # Get ChatGPT portfolio optimization
            chatgpt_portfolio = await self._get_chatgpt_portfolio_optimization(
                symbol, market_data, current_strategies
            )
            
            # Merge portfolio recommendations
            optimized_portfolio = self._merge_portfolio_recommendations(
                manus_portfolio, chatgpt_portfolio, symbol
            )
            
            # Calculate expected performance improvements
            performance_projections = self._calculate_performance_projections(
                optimized_portfolio, market_data
            )
            
            result = {
                'symbol': symbol,
                'timestamp': datetime.utcnow().isoformat(),
                'current_strategies': current_strategies,
                'optimized_portfolio': optimized_portfolio,
                'performance_projections': performance_projections,
                'optimization_rationale': optimized_portfolio.get('rationale', ''),
                'risk_assessment': optimized_portfolio.get('risk_assessment', {}),
                'implementation_priority': optimized_portfolio.get('priority', 'medium')
            }
            
            logger.info(f"Portfolio optimization completed for {symbol}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error optimizing portfolio for {symbol}: {e}")
            return {
                'error': str(e),
                'optimized_portfolio': current_strategies
            }
    
    async def _get_manus_ai_recommendation(
        self, 
        symbol: str, 
        market_data: pd.DataFrame, 
        available_strategies: List[str]
    ) -> AIRecommendation:
        """Get recommendation from Manus AI"""
        try:
            # Get Manus AI strategy suggestions
            manus_result = self.manus_ai.suggest_strategies(symbol, market_data)
            
            # Extract and format strategies
            strategies = []
            if 'recommended_strategies' in manus_result:
                for strategy in manus_result['recommended_strategies'][:5]:  # Top 5
                    if isinstance(strategy, dict):
                        strategies.append({
                            'name': strategy.get('name', ''),
                            'confidence': strategy.get('confidence', 0.5),
                            'priority': strategy.get('priority', 'medium'),
                            'reasoning': strategy.get('reasoning', '')
                        })
            
            # Fallback if no strategies found
            if not strategies:
                strategies = [{'name': s, 'confidence': 0.6, 'priority': 'medium'} 
                            for s in available_strategies[:3]]
            
            return AIRecommendation(
                ai_name='manus_ai',
                strategies=strategies,
                confidence=manus_result.get('confidence', 0.7),
                reasoning=manus_result.get('reasoning', 'Manus AI analysis based on market regime and volatility'),
                timestamp=datetime.utcnow(),
                market_conditions=manus_result.get('market_analysis', {}),
                risk_assessment=manus_result.get('risk_parameters', {})
            )
            
        except Exception as e:
            logger.warning(f"Error getting Manus AI recommendation: {e}")
            return AIRecommendation(
                ai_name='manus_ai',
                strategies=[{'name': s, 'confidence': 0.5} for s in available_strategies[:3]],
                confidence=0.5,
                reasoning='Manus AI analysis unavailable',
                timestamp=datetime.utcnow(),
                market_conditions={},
                risk_assessment={}
            )
    
    async def _get_chatgpt_recommendation(
        self, 
        symbol: str, 
        market_data: pd.DataFrame, 
        available_strategies: List[str],
        news_data: Optional[str] = None
    ) -> AIRecommendation:
        """Get recommendation from ChatGPT"""
        try:
            # Get ChatGPT market analysis
            chatgpt_result = await self.chatgpt_optimizer.analyze_market_conditions(
                symbol, market_data, available_strategies
            )
            
            # Get sentiment analysis if news available
            sentiment_result = None
            if news_data:
                sentiment_result = await self.chatgpt_optimizer.assess_market_sentiment(
                    symbol, news_data, market_data
                )
            
            # Extract strategies from GPT analysis
            strategies = []
            if 'gpt_analysis' in chatgpt_result and 'strategy_ranking' in chatgpt_result['gpt_analysis']:
                for strategy in chatgpt_result['gpt_analysis']['strategy_ranking'][:5]:
                    if isinstance(strategy, dict):
                        strategies.append({
                            'name': strategy.get('name', strategy.get('strategy', '')),
                            'confidence': strategy.get('confidence', strategy.get('score', 0.6)),
                            'priority': strategy.get('priority', 'medium'),
                            'reasoning': strategy.get('reasoning', strategy.get('rationale', ''))
                        })
            
            # Fallback if no strategies found
            if not strategies:
                strategies = [{'name': s, 'confidence': 0.65, 'priority': 'medium'} 
                            for s in available_strategies[:3]]
            
            # Combine market and sentiment analysis
            market_conditions = chatgpt_result.get('market_conditions', {})
            if sentiment_result:
                market_conditions['sentiment'] = sentiment_result.get('sentiment_direction', 'neutral')
                market_conditions['sentiment_confidence'] = sentiment_result.get('confidence_score', 0.5)
            
            return AIRecommendation(
                ai_name='chatgpt',
                strategies=strategies,
                confidence=chatgpt_result.get('confidence_score', 0.7),
                reasoning=chatgpt_result.get('gpt_analysis', {}).get('market_outlook', {}).get('reasoning', 
                                                                    'ChatGPT analysis based on advanced market conditions'),
                timestamp=datetime.utcnow(),
                market_conditions=market_conditions,
                risk_assessment=chatgpt_result.get('risk_assessment', {})
            )
            
        except Exception as e:
            logger.warning(f"Error getting ChatGPT recommendation: {e}")
            return AIRecommendation(
                ai_name='chatgpt',
                strategies=[{'name': s, 'confidence': 0.55} for s in available_strategies[:3]],
                confidence=0.55,
                reasoning='ChatGPT analysis unavailable',
                timestamp=datetime.utcnow(),
                market_conditions={},
                risk_assessment={}
            )
    
    def _calculate_dynamic_weights(self, symbol: str, market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate dynamic weights based on AI performance history"""
        try:
            base_weights = self.ai_weights.copy()
            
            # Adjust based on historical performance
            manus_performance = self.ai_performance_history['manus_ai']['accuracy']
            chatgpt_performance = self.ai_performance_history['chatgpt']['accuracy']
            
            # Performance-based adjustment
            total_performance = manus_performance + chatgpt_performance
            if total_performance > 0:
                performance_factor = 0.2  # Max 20% adjustment
                manus_adj = (manus_performance / total_performance - 0.5) * performance_factor
                chatgpt_adj = (chatgpt_performance / total_performance - 0.5) * performance_factor
                
                base_weights['manus_ai'] += manus_adj
                base_weights['chatgpt'] += chatgpt_adj
            
            # Normalize weights
            total_weight = sum(base_weights.values())
            if total_weight > 0:
                for key in base_weights:
                    base_weights[key] /= total_weight
            
            # Ensure minimum weights
            min_weight = 0.2
            for key in base_weights:
                base_weights[key] = max(min_weight, base_weights[key])
            
            # Re-normalize after minimum enforcement
            total_weight = sum(base_weights.values())
            for key in base_weights:
                base_weights[key] /= total_weight
            
            return base_weights
            
        except Exception as e:
            logger.warning(f"Error calculating dynamic weights: {e}")
            return self.ai_weights.copy()
    
    def _analyze_ai_agreement(
        self, 
        manus_rec: AIRecommendation, 
        chatgpt_rec: AIRecommendation
    ) -> Dict[str, Any]:
        """Analyze agreement between AI recommendations"""
        try:
            # Get strategy names from both recommendations
            manus_strategies = {s.get('name', '') for s in manus_rec.strategies}
            chatgpt_strategies = {s.get('name', '') for s in chatgpt_rec.strategies}
            
            # Calculate strategy overlap
            common_strategies = manus_strategies.intersection(chatgpt_strategies)
            total_unique_strategies = manus_strategies.union(chatgpt_strategies)
            
            strategy_agreement = len(common_strategies) / len(total_unique_strategies) if total_unique_strategies else 0
            
            # Calculate confidence agreement
            confidence_diff = abs(manus_rec.confidence - chatgpt_rec.confidence)
            confidence_agreement = 1.0 - min(confidence_diff, 1.0)
            
            # Overall agreement score (weighted average)
            agreement_score = (strategy_agreement * 0.7) + (confidence_agreement * 0.3)
            
            # Identify conflict areas
            conflict_areas = []
            if strategy_agreement < 0.5:
                conflict_areas.append('strategy_selection')
            if confidence_diff > 0.3:
                conflict_areas.append('confidence_levels')
            if abs(len(manus_rec.strategies) - len(chatgpt_rec.strategies)) > 2:
                conflict_areas.append('recommendation_count')
            
            return {
                'agreement_score': agreement_score,
                'strategy_agreement': strategy_agreement,
                'confidence_agreement': confidence_agreement,
                'common_strategies': list(common_strategies),
                'conflict_areas': conflict_areas,
                'manus_unique': list(manus_strategies - chatgpt_strategies),
                'chatgpt_unique': list(chatgpt_strategies - manus_strategies)
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing AI agreement: {e}")
            return {
                'agreement_score': 0.5,
                'strategy_agreement': 0.5,
                'confidence_agreement': 0.5,
                'common_strategies': [],
                'conflict_areas': ['analysis_error'],
                'manus_unique': [],
                'chatgpt_unique': []
            }
    
    def _determine_consensus_level(self, agreement_score: float) -> ConsensusLevel:
        """Determine consensus level based on agreement score"""
        if agreement_score >= self.consensus_thresholds['high_agreement']:
            return ConsensusLevel.HIGH_AGREEMENT
        elif agreement_score >= self.consensus_thresholds['moderate_agreement']:
            return ConsensusLevel.MODERATE_AGREEMENT
        elif agreement_score >= self.consensus_thresholds['low_agreement']:
            return ConsensusLevel.LOW_AGREEMENT
        else:
            return ConsensusLevel.DISAGREEMENT
    
    def _resolve_conflicts(
        self, 
        manus_rec: AIRecommendation, 
        chatgpt_rec: AIRecommendation,
        weights: Dict[str, float],
        symbol: str
    ) -> Dict[str, Any]:
        """Resolve conflicts between AI recommendations"""
        try:
            # Choose resolution method based on conflict type
            resolution_method = 'hybrid_ensemble'  # Default to most robust method
            
            if resolution_method == 'hybrid_ensemble':
                # Create ensemble of both recommendations with weighting
                merged_strategies = []
                
                # Process strategies from both AIs
                all_strategies = {}
                
                # Add Manus AI strategies with weights
                for strategy in manus_rec.strategies:
                    name = strategy.get('name', '')
                    if name:
                        all_strategies[name] = {
                            'name': name,
                            'manus_confidence': strategy.get('confidence', 0.5),
                            'chatgpt_confidence': 0.0,
                            'combined_score': strategy.get('confidence', 0.5) * weights['manus_ai'],
                            'sources': ['manus_ai']
                        }
                
                # Add/update with ChatGPT strategies
                for strategy in chatgpt_rec.strategies:
                    name = strategy.get('name', '')
                    if name:
                        if name in all_strategies:
                            # Strategy appears in both - boost confidence
                            all_strategies[name]['chatgpt_confidence'] = strategy.get('confidence', 0.5)
                            all_strategies[name]['combined_score'] = (
                                all_strategies[name]['manus_confidence'] * weights['manus_ai'] +
                                strategy.get('confidence', 0.5) * weights['chatgpt']
                            )
                            all_strategies[name]['sources'].append('chatgpt')
                        else:
                            # New strategy from ChatGPT
                            all_strategies[name] = {
                                'name': name,
                                'manus_confidence': 0.0,
                                'chatgpt_confidence': strategy.get('confidence', 0.5),
                                'combined_score': strategy.get('confidence', 0.5) * weights['chatgpt'],
                                'sources': ['chatgpt']
                            }
                
                # Sort by combined score and select top strategies
                sorted_strategies = sorted(
                    all_strategies.values(), 
                    key=lambda x: x['combined_score'], 
                    reverse=True
                )
                
                # Format final strategies
                for strategy in sorted_strategies[:5]:  # Top 5
                    merged_strategies.append({
                        'name': strategy['name'],
                        'confidence': strategy['combined_score'],
                        'sources': strategy['sources'],
                        'manus_confidence': strategy['manus_confidence'],
                        'chatgpt_confidence': strategy['chatgpt_confidence'],
                        'consensus_method': 'hybrid_ensemble'
                    })
            
            return {
                'method': resolution_method,
                'strategies': merged_strategies,
                'resolution_confidence': 0.8
            }
            
        except Exception as e:
            logger.warning(f"Error resolving conflicts: {e}")
            # Fallback to simple weighted average
            return {
                'method': 'fallback_weighted',
                'strategies': manus_rec.strategies[:3] if weights['manus_ai'] > weights['chatgpt'] 
                            else chatgpt_rec.strategies[:3],
                'resolution_confidence': 0.5
            }
    
    def _merge_agreements(
        self, 
        manus_rec: AIRecommendation, 
        chatgpt_rec: AIRecommendation,
        weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """Merge recommendations when AIs are in agreement"""
        try:
            # Get common strategies
            manus_strategy_names = {s.get('name', '') for s in manus_rec.strategies}
            chatgpt_strategy_names = {s.get('name', '') for s in chatgpt_rec.strategies}
            common_strategies = manus_strategy_names.intersection(chatgpt_strategy_names)
            
            merged_strategies = []
            
            # Process common strategies first (highest confidence)
            for strategy_name in common_strategies:
                manus_strategy = next((s for s in manus_rec.strategies if s.get('name') == strategy_name), None)
                chatgpt_strategy = next((s for s in chatgpt_rec.strategies if s.get('name') == strategy_name), None)
                
                if manus_strategy and chatgpt_strategy:
                    # Weighted average of confidences
                    combined_confidence = (
                        manus_strategy.get('confidence', 0.5) * weights['manus_ai'] +
                        chatgpt_strategy.get('confidence', 0.5) * weights['chatgpt']
                    )
                    
                    merged_strategies.append({
                        'name': strategy_name,
                        'confidence': combined_confidence,
                        'sources': ['manus_ai', 'chatgpt'],
                        'agreement_boost': 0.1,  # Boost for agreement
                        'manus_confidence': manus_strategy.get('confidence', 0.5),
                        'chatgpt_confidence': chatgpt_strategy.get('confidence', 0.5)
                    })
            
            # Add remaining strategies from both AIs
            all_remaining = []
            
            for strategy in manus_rec.strategies:
                if strategy.get('name') not in common_strategies:
                    all_remaining.append({
                        'name': strategy.get('name', ''),
                        'confidence': strategy.get('confidence', 0.5) * weights['manus_ai'],
                        'sources': ['manus_ai'],
                        'manus_confidence': strategy.get('confidence', 0.5),
                        'chatgpt_confidence': 0.0
                    })
            
            for strategy in chatgpt_rec.strategies:
                if strategy.get('name') not in common_strategies:
                    all_remaining.append({
                        'name': strategy.get('name', ''),
                        'confidence': strategy.get('confidence', 0.5) * weights['chatgpt'],
                        'sources': ['chatgpt'],
                        'manus_confidence': 0.0,
                        'chatgpt_confidence': strategy.get('confidence', 0.5)
                    })
            
            # Sort remaining by confidence and add top ones
            all_remaining.sort(key=lambda x: x['confidence'], reverse=True)
            merged_strategies.extend(all_remaining[:5 - len(merged_strategies)])
            
            return {
                'method': 'agreement_merge',
                'strategies': merged_strategies,
                'resolution_confidence': 0.9
            }
            
        except Exception as e:
            logger.warning(f"Error merging agreements: {e}")
            return {
                'method': 'simple_merge',
                'strategies': (manus_rec.strategies + chatgpt_rec.strategies)[:5],
                'resolution_confidence': 0.6
            }
    
    def _calculate_overall_confidence(
        self, 
        manus_rec: AIRecommendation, 
        chatgpt_rec: AIRecommendation,
        agreement_analysis: Dict[str, Any],
        weights: Dict[str, float]
    ) -> float:
        """Calculate overall confidence for the consensus"""
        try:
            # Base confidence from weighted average
            base_confidence = (
                manus_rec.confidence * weights['manus_ai'] +
                chatgpt_rec.confidence * weights['chatgpt']
            )
            
            # Agreement bonus
            agreement_bonus = agreement_analysis['agreement_score'] * 0.15  # Up to 15% bonus
            
            # Conflict penalty
            conflict_penalty = len(agreement_analysis['conflict_areas']) * 0.05  # 5% per conflict area
            
            # Final confidence
            final_confidence = base_confidence + agreement_bonus - conflict_penalty
            
            # Ensure confidence is within bounds
            return max(0.0, min(1.0, final_confidence))
            
        except Exception as e:
            logger.warning(f"Error calculating overall confidence: {e}")
            return 0.6  # Safe default
    
    def _generate_consensus_reasoning(
        self,
        manus_rec: AIRecommendation,
        chatgpt_rec: AIRecommendation,
        agreement_analysis: Dict[str, Any],
        consensus_level: ConsensusLevel,
        resolution_result: Dict[str, Any]
    ) -> str:
        """Generate human-readable reasoning for the consensus decision"""
        try:
            reasoning_parts = []
            
            # Consensus level explanation
            if consensus_level == ConsensusLevel.HIGH_AGREEMENT:
                reasoning_parts.append("Both AI systems show strong agreement on strategy recommendations.")
            elif consensus_level == ConsensusLevel.MODERATE_AGREEMENT:
                reasoning_parts.append("AI systems show moderate agreement with some minor differences.")
            elif consensus_level == ConsensusLevel.LOW_AGREEMENT:
                reasoning_parts.append("AI systems have limited agreement, requiring careful conflict resolution.")
            else:
                reasoning_parts.append("AI systems disagree significantly, requiring advanced consensus algorithms.")
            
            # Common strategies
            if agreement_analysis['common_strategies']:
                reasoning_parts.append(
                    f"Common recommended strategies: {', '.join(agreement_analysis['common_strategies'][:3])}"
                )
            
            # Unique insights
            if agreement_analysis['manus_unique']:
                reasoning_parts.append(
                    f"Manus AI uniquely recommends: {', '.join(agreement_analysis['manus_unique'][:2])}"
                )
            
            if agreement_analysis['chatgpt_unique']:
                reasoning_parts.append(
                    f"ChatGPT uniquely recommends: {', '.join(agreement_analysis['chatgpt_unique'][:2])}"
                )
            
            # Resolution method
            reasoning_parts.append(f"Final recommendation uses {resolution_result['method']} approach.")
            
            return " ".join(reasoning_parts)
            
        except Exception as e:
            logger.warning(f"Error generating consensus reasoning: {e}")
            return "Consensus recommendation generated using dual-AI analysis."
    
    def _update_performance_tracking(self, symbol: str, consensus_result: ConsensusResult):
        """Update AI performance tracking"""
        try:
            # This would typically update a database with historical performance
            # For now, we'll update in-memory tracking
            
            timestamp = datetime.utcnow()
            
            # Track consensus decisions for later evaluation
            performance_entry = {
                'symbol': symbol,
                'timestamp': timestamp,
                'consensus_level': consensus_result.consensus_level.value,
                'overall_confidence': consensus_result.overall_confidence,
                'strategies': consensus_result.recommended_strategies,
                'ai_contributions': consensus_result.ai_contributions
            }
            
            # Add to recent performance tracking
            for ai_name in ['manus_ai', 'chatgpt']:
                if ai_name in consensus_result.ai_contributions:
                    self.ai_performance_history[ai_name]['recent_performance'].append(performance_entry)
                    
                    # Keep only last 100 entries
                    if len(self.ai_performance_history[ai_name]['recent_performance']) > 100:
                        self.ai_performance_history[ai_name]['recent_performance'].pop(0)
            
            logger.debug(f"Performance tracking updated for {symbol}")
            
        except Exception as e:
            logger.warning(f"Error updating performance tracking: {e}")
    
    def _fallback_consensus(self, symbol: str, available_strategies: List[str]) -> ConsensusResult:
        """Fallback consensus when AI systems fail"""
        return ConsensusResult(
            recommended_strategies=[
                {'name': strategy, 'confidence': 0.5} 
                for strategy in available_strategies[:3]
            ],
            consensus_level=ConsensusLevel.LOW_AGREEMENT,
            overall_confidence=0.4,
            agreement_score=0.3,
            conflict_areas=['ai_system_error'],
            resolution_method='fallback',
            ai_contributions={
                'manus_ai': {'weight': 0.5, 'confidence': 0.4},
                'chatgpt': {'weight': 0.5, 'confidence': 0.4}
            },
            reasoning='Fallback recommendation due to AI system unavailability',
            timestamp=datetime.utcnow()
        )
    
    # Additional helper methods for validation and optimization
    async def _validate_with_manus_ai(self, symbol: str, strategy_name: str, 
                                    market_data: pd.DataFrame, strategy_params: Dict) -> Dict:
        """Validate strategy with Manus AI"""
        try:
            # Use Manus AI's strategy suggestion method to validate
            result = self.manus_ai.suggest_strategies(symbol, market_data)
            
            # Check if the strategy is recommended
            recommended_strategies = [s.get('name', '') for s in result.get('recommended_strategies', [])]
            is_recommended = strategy_name in recommended_strategies
            
            return {
                'recommended': is_recommended,
                'confidence': 0.7 if is_recommended else 0.3,
                'reasoning': f"Manus AI {'supports' if is_recommended else 'does not support'} {strategy_name}"
            }
            
        except Exception as e:
            return {'recommended': False, 'confidence': 0.0, 'error': str(e)}
    
    async def _validate_with_chatgpt(self, symbol: str, strategy_name: str, 
                                   market_data: pd.DataFrame, strategy_params: Dict) -> Dict:
        """Validate strategy with ChatGPT"""
        try:
            # Use ChatGPT's analysis method to validate
            result = await self.chatgpt_optimizer.analyze_market_conditions(
                symbol, market_data, [strategy_name]
            )
            
            # Check if strategy is in recommendations
            recommended_strategies = []
            if 'gpt_analysis' in result and 'strategy_ranking' in result['gpt_analysis']:
                recommended_strategies = [s.get('name', '') for s in result['gpt_analysis']['strategy_ranking']]
            
            is_recommended = strategy_name in recommended_strategies
            
            return {
                'recommended': is_recommended,
                'confidence': result.get('confidence_score', 0.5),
                'reasoning': f"ChatGPT {'supports' if is_recommended else 'does not support'} {strategy_name}"
            }
            
        except Exception as e:
            return {'recommended': False, 'confidence': 0.0, 'error': str(e)}
    
    def _compare_validations(self, manus_validation: Dict, chatgpt_validation: Dict) -> Dict:
        """Compare validation results from both AIs"""
        manus_rec = manus_validation.get('recommended', False)
        chatgpt_rec = chatgpt_validation.get('recommended', False)
        
        if manus_rec and chatgpt_rec:
            agreement = 'strong_positive'
        elif not manus_rec and not chatgpt_rec:
            agreement = 'strong_negative'
        else:
            agreement = 'disagreement'
        
        return {
            'agreement': agreement,
            'manus_recommended': manus_rec,
            'chatgpt_recommended': chatgpt_rec,
            'confidence_difference': abs(
                manus_validation.get('confidence', 0) - 
                chatgpt_validation.get('confidence', 0)
            )
        }
    
    def _create_consensus_validation(self, manus_validation: Dict, 
                                   chatgpt_validation: Dict, 
                                   validation_agreement: Dict) -> Dict:
        """Create consensus validation result"""
        agreement = validation_agreement['agreement']
        
        if agreement == 'strong_positive':
            recommendation = 'strongly_recommended'
            confidence = (manus_validation.get('confidence', 0) + 
                         chatgpt_validation.get('confidence', 0)) / 2 + 0.1  # Agreement bonus
        elif agreement == 'strong_negative':
            recommendation = 'not_recommended'
            confidence = (manus_validation.get('confidence', 0) + 
                         chatgpt_validation.get('confidence', 0)) / 2 + 0.1  # Agreement bonus
        else:
            recommendation = 'inconclusive'
            confidence = (manus_validation.get('confidence', 0) + 
                         chatgpt_validation.get('confidence', 0)) / 2 - 0.1  # Disagreement penalty
        
        return {
            'recommendation': recommendation,
            'confidence': max(0.0, min(1.0, confidence)),
            'agreement_type': agreement
        }
    
    async def _get_manus_portfolio_optimization(self, symbol: str, market_data: pd.DataFrame, 
                                              current_strategies: List[Dict]) -> Dict:
        """Get portfolio optimization from Manus AI"""
        try:
            result = self.manus_ai.suggest_strategies(symbol, market_data)
            return {
                'optimized_strategies': result.get('recommended_strategies', current_strategies),
                'confidence': result.get('confidence', 0.6),
                'source': 'manus_ai'
            }
        except Exception as e:
            return {'optimized_strategies': current_strategies, 'confidence': 0.0, 'error': str(e)}
    
    async def _get_chatgpt_portfolio_optimization(self, symbol: str, market_data: pd.DataFrame, 
                                                current_strategies: List[Dict]) -> Dict:
        """Get portfolio optimization from ChatGPT"""
        try:
            # Use available strategies as input
            strategy_names = [s.get('name', '') for s in current_strategies]
            result = await self.chatgpt_optimizer.analyze_market_conditions(
                symbol, market_data, strategy_names
            )
            
            optimized_strategies = []
            if 'gpt_analysis' in result and 'strategy_ranking' in result['gpt_analysis']:
                optimized_strategies = result['gpt_analysis']['strategy_ranking']
            
            return {
                'optimized_strategies': optimized_strategies or current_strategies,
                'confidence': result.get('confidence_score', 0.6),
                'source': 'chatgpt'
            }
        except Exception as e:
            return {'optimized_strategies': current_strategies, 'confidence': 0.0, 'error': str(e)}
    
    def _merge_portfolio_recommendations(self, manus_portfolio: Dict, 
                                       chatgpt_portfolio: Dict, symbol: str) -> Dict:
        """Merge portfolio recommendations from both AIs"""
        try:
            manus_strategies = manus_portfolio.get('optimized_strategies', [])
            chatgpt_strategies = chatgpt_portfolio.get('optimized_strategies', [])
            
            # Combine and deduplicate strategies
            all_strategies = {}
            
            # Add Manus strategies
            for i, strategy in enumerate(manus_strategies[:5]):
                name = strategy.get('name', '')
                if name:
                    all_strategies[name] = {
                        'name': name,
                        'manus_rank': i + 1,
                        'chatgpt_rank': None,
                        'combined_score': (6 - (i + 1)) * 0.5  # Higher rank = higher score
                    }
            
            # Add/update with ChatGPT strategies
            for i, strategy in enumerate(chatgpt_strategies[:5]):
                name = strategy.get('name', '')
                if name:
                    if name in all_strategies:
                        all_strategies[name]['chatgpt_rank'] = i + 1
                        all_strategies[name]['combined_score'] += (6 - (i + 1)) * 0.5
                    else:
                        all_strategies[name] = {
                            'name': name,
                            'manus_rank': None,
                            'chatgpt_rank': i + 1,
                            'combined_score': (6 - (i + 1)) * 0.5
                        }
            
            # Sort by combined score
            sorted_strategies = sorted(
                all_strategies.values(),
                key=lambda x: x['combined_score'],
                reverse=True
            )
            
            return {
                'optimized_strategies': sorted_strategies[:5],
                'rationale': 'Combined optimization from Manus AI and ChatGPT',
                'confidence': (manus_portfolio.get('confidence', 0) + 
                             chatgpt_portfolio.get('confidence', 0)) / 2
            }
            
        except Exception as e:
            logger.warning(f"Error merging portfolio recommendations: {e}")
            return {
                'optimized_strategies': manus_portfolio.get('optimized_strategies', []),
                'rationale': 'Fallback to Manus AI recommendations',
                'confidence': 0.5
            }
    
    def _calculate_performance_projections(self, optimized_portfolio: Dict, 
                                         market_data: pd.DataFrame) -> Dict:
        """Calculate expected performance improvements"""
        try:
            # Simple projection based on strategy count and confidence
            strategy_count = len(optimized_portfolio.get('optimized_strategies', []))
            confidence = optimized_portfolio.get('confidence', 0.5)
            
            # Estimate improvements (placeholder for actual backtesting)
            expected_return_improvement = strategy_count * confidence * 0.02  # 2% per strategy
            expected_sharpe_improvement = confidence * 0.1  # 10% sharpe improvement
            expected_drawdown_reduction = confidence * 0.05  # 5% drawdown reduction
            
            return {
                'expected_return_improvement': f"{expected_return_improvement:.1%}",
                'expected_sharpe_improvement': f"+{expected_sharpe_improvement:.2f}",
                'expected_drawdown_reduction': f"{expected_drawdown_reduction:.1%}",
                'confidence_level': confidence
            }
            
        except Exception as e:
            logger.warning(f"Error calculating performance projections: {e}")
            return {
                'expected_return_improvement': "0.0%",
                'expected_sharpe_improvement': "+0.00",
                'expected_drawdown_reduction': "0.0%",
                'confidence_level': 0.5
            }

# Export main classes
__all__ = ['AIStrategyConsensus', 'ConsensusResult', 'AIRecommendation', 'ConsensusLevel']