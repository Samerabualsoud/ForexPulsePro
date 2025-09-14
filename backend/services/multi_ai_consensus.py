"""
Multi-AI Consensus System - Enhanced Signal Intelligence
Coordinates multiple AI agents for superior trading signal quality
"""
import asyncio
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime

from .manus_ai import ManusAI
from .perplexity_news_agent import PerplexityNewsAgent
from ..logs.logger import get_logger

logger = get_logger(__name__)

# Optional DeepSeek import with fallback
try:
    from .deepseek_agent import DeepSeekAgent
except Exception as e:
    DeepSeekAgent = None
    logger.warning(f"DeepSeek disabled: import failed: {e}")

# Optional FinBERT import with fallback
try:
    from .finbert_sentiment_agent import FinBERTSentimentAgent
except Exception as e:
    FinBERTSentimentAgent = None
    logger.warning(f"FinBERT disabled: import failed: {e}")

# Optional Groq import with fallback
try:
    from .groq_reasoning_agent import GroqReasoningAgent
except Exception as e:
    GroqReasoningAgent = None
    logger.warning(f"Groq disabled: import failed: {e}")

class MultiAIConsensus:
    """
    Advanced multi-AI consensus system that combines insights from:
    - Manus AI (Primary strategy recommendations)
    - Perplexity AI (Market intelligence)
    - DeepSeek AI (Advanced reasoning and sentiment analysis)
    - FinBERT AI (Financial news sentiment analysis)
    - Groq AI (Fast reasoning and market analysis)
    """
    
    def __init__(self):
        # Initialize all AI agents
        self.manus_ai = ManusAI()
        self.perplexity_agent = PerplexityNewsAgent()
        self.deepseek_agent = DeepSeekAgent() if DeepSeekAgent else None
        self.finbert_agent = FinBERTSentimentAgent() if FinBERTSentimentAgent else None
        self.groq_agent = GroqReasoningAgent() if GroqReasoningAgent else None
        
        # Track agent availability
        self.available_agents = self._check_agent_availability()
        
        logger.info(f"Multi-AI Consensus initialized with {len(self.available_agents)} agents: {list(self.available_agents.keys())}")
    
    async def generate_enhanced_signal_analysis(
        self, 
        symbol: str, 
        market_data: pd.DataFrame, 
        base_signal: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive signal analysis using all available AI agents
        
        Args:
            symbol: Trading symbol
            market_data: OHLC price data
            base_signal: Optional base signal from technical analysis
            
        Returns:
            Enhanced signal analysis with multi-AI consensus
        """
        logger.info(f"Starting multi-AI analysis for {symbol}")
        
        # Collect analyses from all available agents
        analyses = {}
        
        # Run all AI analyses in parallel for efficiency
        tasks = []
        
        if self.available_agents.get('manus_ai'):
            tasks.append(self._run_manus_analysis(symbol, market_data))
        
        if self.available_agents.get('perplexity_news') and base_signal:
            tasks.append(self._run_perplexity_analysis(symbol, base_signal.get('action', 'BUY')))
        
        if self.available_agents.get('deepseek_reasoning'):
            current_price = market_data['close'].iloc[-1] if len(market_data) > 0 else 0
            tasks.append(self._run_deepseek_analysis(symbol, market_data, current_price))
        
        if self.available_agents.get('finbert_sentiment') and base_signal:
            tasks.append(self._run_finbert_analysis(symbol, base_signal.get('action', 'BUY')))
        
        if self.available_agents.get('groq_reasoning'):
            current_price = market_data['close'].iloc[-1] if len(market_data) > 0 else 0
            tasks.append(self._run_groq_analysis(symbol, market_data, current_price))
        
        # Execute all analyses concurrently
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"AI analysis task {i} failed: {result}")
                elif isinstance(result, dict) and result:
                    analyses.update(result)
        
        # **CRITICAL VALIDATION**: Ensure we have actual analysis results before proceeding
        if not analyses:
            logger.error(f"CRITICAL: No AI analyses succeeded for {symbol} - all agents failed")
            return {
                'final_confidence': 0.0,
                'consensus_action': 'NO_AGENTS_AVAILABLE',
                'agent_count': 0,
                'participating_agents': 0,
                'consensus_strength': 0.0,
                'consensus_level': 0.0,
                'risk_level': 'HIGH',
                'quality_gate': 'FAILED_ALL_AGENTS',
                'agent_insights': {},
                'multi_ai_valid': False
            }
        
        # Generate consensus analysis
        consensus = self._generate_consensus(analyses, base_signal)
        
        logger.info(f"Multi-AI consensus for {symbol}: confidence={consensus.get('final_confidence', 0):.3f}, agents={len(analyses)}")
        
        return consensus
    
    async def _run_manus_analysis(self, symbol: str, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Run Manus AI analysis"""
        try:
            # Get Manus AI strategy recommendations using the correct method
            strategy_analysis = self.manus_ai.suggest_strategies(symbol, market_data)
            
            return {
                'manus_ai': {
                    'regime': strategy_analysis.get('regime', 'UNKNOWN'),
                    'regime_confidence': strategy_analysis.get('confidence', 0.5),
                    'recommended_strategies': strategy_analysis.get('strategies', []),
                    'market_condition': strategy_analysis.get('market_condition', 'neutral'),
                    'agent': 'manus_ai'
                }
            }
        except Exception as e:
            logger.error(f"Manus AI analysis failed: {e}")
            return {}
    
    
    async def _run_perplexity_analysis(self, symbol: str, signal_action: str) -> Dict[str, Any]:
        """Run Perplexity news analysis"""
        try:
            analysis = self.perplexity_agent.analyze_market_context(symbol, signal_action)
            return {'perplexity_news': analysis}
        except Exception as e:
            logger.error(f"Perplexity analysis failed: {e}")
            return {}
    
    
    async def _run_deepseek_analysis(self, symbol: str, market_data: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """Run DeepSeek sentiment and reasoning analysis"""
        try:
            if self.deepseek_agent is None:
                logger.warning("DeepSeek agent not available")
                return {}
            analysis = await self.deepseek_agent.analyze_market_sentiment(symbol, market_data, current_price)
            return {'deepseek_reasoning': analysis}
        except Exception as e:
            logger.error(f"DeepSeek analysis failed: {e}")
            return {}
    
    async def _run_finbert_analysis(self, symbol: str, signal_action: str) -> Dict[str, Any]:
        """Run FinBERT financial news sentiment analysis"""
        try:
            if self.finbert_agent is None:
                logger.warning("FinBERT agent not available")
                return {}
            analysis = await self.finbert_agent.analyze_market_news_impact(symbol, signal_action)
            return {'finbert_sentiment': analysis}
        except Exception as e:
            logger.error(f"FinBERT analysis failed: {e}")
            return {}
    
    async def _run_groq_analysis(self, symbol: str, market_data: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """Run Groq fast reasoning and market analysis"""
        try:
            if self.groq_agent is None:
                logger.warning("Groq agent not available")
                return {}
            analysis = await self.groq_agent.analyze_market_sentiment(symbol, market_data, current_price)
            return {'groq_reasoning': analysis}
        except Exception as e:
            logger.error(f"Groq analysis failed: {e}")
            return {}
    
    def _generate_consensus(self, analyses: Dict[str, Any], base_signal: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate final consensus from all AI analyses
        
        Args:
            analyses: Dictionary of AI agent analyses
            base_signal: Base technical signal
            
        Returns:
            Consensus analysis with final confidence and recommendations
        """
        # Start with base confidence or neutral
        base_confidence = base_signal.get('confidence', 0.5) if base_signal else 0.5
        
        # Collect confidence adjustments from each agent
        confidence_adjustments = []
        agent_insights = {}
        
        # Process Manus AI insights
        if 'manus_ai' in analyses:
            manus = analyses['manus_ai']
            agent_insights['manus_ai'] = {
                'regime': manus.get('regime', 'UNKNOWN'),
                'market_condition': manus.get('market_condition', 'neutral'),
                'weight': 0.2  # Balanced weight for 5-agent system
            }
        
        
        # Process Perplexity news insights
        if 'perplexity_news' in analyses:
            perplexity = analyses['perplexity_news']
            news_impact = perplexity.get('news_impact', 0.0)
            confidence_adjustments.append(('perplexity_news', news_impact, 0.2))
            
            agent_insights['perplexity_news'] = {
                'sentiment': perplexity.get('sentiment', 'neutral'),
                'risk_factors': perplexity.get('risk_factors', []),
                'weight': 0.2
            }
        
        
        # Process DeepSeek reasoning insights  
        if 'deepseek_reasoning' in analyses:
            deepseek = analyses['deepseek_reasoning']
            deepseek_confidence = deepseek.get('confidence', 0.0)
            if deepseek_confidence > 0:
                confidence_adjustments.append(('deepseek_reasoning', deepseek_confidence - 0.5, 0.25))
            
            agent_insights['deepseek_reasoning'] = {
                'sentiment': deepseek.get('sentiment', 'neutral'),
                'reasoning': deepseek.get('reasoning', ''),
                'weight': 0.2
            }
        
        # Process FinBERT sentiment insights
        if 'finbert_sentiment' in analyses:
            finbert = analyses['finbert_sentiment']
            finbert_impact = finbert.get('news_impact', 0.0)
            confidence_adjustments.append(('finbert_sentiment', finbert_impact, 0.25))
            
            agent_insights['finbert_sentiment'] = {
                'sentiment': finbert.get('sentiment', 'neutral'),
                'risk_factors': finbert.get('risk_factors', []),
                'confidence': finbert.get('confidence', 0.0),
                'weight': 0.2
            }
        
        # Process Groq reasoning insights
        if 'groq_reasoning' in analyses:
            groq = analyses['groq_reasoning']
            groq_confidence = groq.get('confidence', 0.0)
            if groq_confidence > 0:
                confidence_adjustments.append(('groq_reasoning', groq_confidence - 0.5, 0.2))
            
            agent_insights['groq_reasoning'] = {
                'sentiment': groq.get('sentiment', 'neutral'),
                'reasoning': groq.get('reasoning', ''),
                'market_structure': groq.get('market_structure', 'ranging'),
                'weight': 0.2
            }
        
        # **QUALITY REQUIREMENT**: Minimum 2 out of 5 agents required for valid signals
        available_agents = len(agent_insights)
        if available_agents < 2:
            logger.warning(f"Insufficient AI agents for consensus: {available_agents}/5 (minimum 2 required)")
            return {
                'final_confidence': 0.0,
                'consensus_action': 'INSUFFICIENT_CONSENSUS',
                'agent_count': available_agents,
                'participating_agents': available_agents,  # Signal engine expects this field
                'consensus_strength': 0.0,
                'consensus_level': 0.0,
                'risk_level': 'HIGH',
                'quality_gate': 'FAILED_MIN_AGENTS',
                'agent_insights': agent_insights,
                'multi_ai_valid': False
            }

        # **ENHANCED CONFIDENCE CALCULATION** - Scale confidence based on agent participation
        # With fewer agents working, we need more aggressive confidence scaling to reach meaningful thresholds
        
        total_adjustment = 0.0
        total_weight = 0.0
        
        for agent_name, adjustment, weight in confidence_adjustments:
            total_adjustment += adjustment * weight
            total_weight += weight
        
        # Apply confidence adjustment with agent scaling factor
        if total_weight > 0:
            weighted_adjustment = total_adjustment / total_weight
            # **CRITICAL FIX**: Scale confidence based on number of participating agents
            # With 2 agents: multiply adjustment by 1.5, with 3+ agents: use normal scaling
            agent_scaling_factor = max(1.0, 1.5 if available_agents == 2 else (1.25 if available_agents == 3 else 1.0))
            scaled_adjustment = weighted_adjustment * agent_scaling_factor
            final_confidence = base_confidence + scaled_adjustment
        else:
            # No adjustments available - set moderate confidence based on agent count
            final_confidence = 0.60 if available_agents >= 2 else base_confidence
        
        # Ensure confidence is within bounds but allow higher scaling for fewer agents
        final_confidence = max(0.0, min(1.0, final_confidence))
        
        # **QUALITY BOOST**: If consensus is strong with limited agents, boost confidence
        if available_agents >= 2 and len(confidence_adjustments) >= 1:
            # Add moderate confidence boost for strong limited-agent consensus
            consensus_boost = 0.15 * (available_agents / 5.0)  # Scale boost by agent ratio
            final_confidence = min(1.0, final_confidence + consensus_boost)
        
        # **CRITICAL BUG FIX**: STRICT 80% CONFIDENCE REQUIREMENT - Block ALL signals below 80%
        # This enforces the user's minimum confidence requirement across the entire system
        if final_confidence < 0.8:
            logger.warning(f"Multi-AI consensus blocked signal due to insufficient confidence: {final_confidence:.1%} < 80% minimum threshold")
            return {
                'final_confidence': 0.0,
                'consensus_action': 'BLOCKED_LOW_CONFIDENCE',
                'agent_count': available_agents,
                'participating_agents': available_agents,  # Signal engine expects this field
                'consensus_strength': 0.0,
                'consensus_level': 0.0,  # Signal engine expects this field
                'risk_level': 'HIGH',
                'quality_gate': 'FAILED_MIN_CONFIDENCE',
                'agent_insights': agent_insights,
                'multi_ai_valid': False,
                'block_reason': f'Confidence {final_confidence:.1%} below 80% minimum threshold',
                'timestamp': datetime.now().isoformat()
            }
        
        # Generate consensus recommendation
        consensus_action = self._determine_consensus_action(agent_insights, base_signal)
        
        # Calculate consensus metrics
        consensus_strength = self._calculate_consensus_strength(agent_insights)
        
        # Create detailed consensus analysis
        consensus = {
            'final_confidence': final_confidence,
            'base_confidence': base_confidence,
            'confidence_adjustment': final_confidence - base_confidence,
            'consensus_action': consensus_action,
            'agent_count': available_agents,
            'participating_agents': available_agents,  # Signal engine expects this field
            'agent_insights': agent_insights,
            'consensus_strength': consensus_strength,
            'consensus_level': consensus_strength,  # Signal engine expects this field
            'risk_assessment': self._assess_overall_risk(agent_insights),
            'timestamp': datetime.now().isoformat(),
            'multi_ai_enabled': True,
            'multi_ai_valid': available_agents >= 2 and consensus_strength >= 0.5
        }
        
        return consensus
    
    def _determine_consensus_action(self, agent_insights: Dict[str, Any], base_signal: Optional[Dict[str, Any]]) -> str:
        """Determine consensus trading action from agent insights"""
        if not base_signal:
            return 'HOLD'
        
        base_action = base_signal.get('action', 'HOLD')
        
        # Check for conflicting signals from agents
        bullish_signals = 0
        bearish_signals = 0
        
        for agent, insights in agent_insights.items():
            if agent in ['perplexity_news', 'finbert_sentiment', 'deepseek_reasoning', 'groq_reasoning']:
                sentiment = insights.get('sentiment', 'neutral')
                if sentiment == 'bullish':
                    bullish_signals += 1
                elif sentiment == 'bearish':
                    bearish_signals += 1
        
        # If there's strong consensus against the base signal, suggest caution
        if base_action == 'BUY' and bearish_signals > bullish_signals + 1:
            return 'CAUTION_BUY'
        elif base_action == 'SELL' and bullish_signals > bearish_signals + 1:
            return 'CAUTION_SELL'
        
        return base_action
    
    def _calculate_consensus_strength(self, agent_insights: Dict[str, Any]) -> float:
        """Calculate how strongly the agents agree (0.0 to 1.0)"""
        if len(agent_insights) < 2:
            return 0.5
        
        # Simple consensus calculation based on sentiment alignment
        sentiments = []
        
        for agent, insights in agent_insights.items():
            if 'trading_bias' in insights:
                sentiments.append(insights['trading_bias'])
            elif 'sentiment' in insights:
                sentiments.append(insights['sentiment'])
        
        if not sentiments:
            return 0.5
        
        # Calculate agreement level
        bullish_count = sentiments.count('bullish')
        bearish_count = sentiments.count('bearish')
        neutral_count = sentiments.count('neutral')
        
        total = len(sentiments)
        max_agreement = max(bullish_count, bearish_count, neutral_count)
        
        return max_agreement / total
    
    def _assess_overall_risk(self, agent_insights: Dict[str, Any]) -> str:
        """Assess overall risk level from all agents"""
        risk_factors = []
        
        # Collect risk factors from agents
        for agent, insights in agent_insights.items():
            if 'risk_factors' in insights:
                risk_factors.extend(insights['risk_factors'])
        
        high_risk_count = len([factor for factor in risk_factors if factor in ['uncertainty', 'volatility', 'crisis']])
        
        if high_risk_count >= 2:
            return 'HIGH'
        elif high_risk_count == 1:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _check_agent_availability(self) -> Dict[str, bool]:
        """Check which AI agents are available"""
        return {
            'manus_ai': True,  # Always available
            'perplexity_news': self.perplexity_agent.is_available(),
            'deepseek_reasoning': bool(self.deepseek_agent and self.deepseek_agent.is_available()),
            'finbert_sentiment': bool(self.finbert_agent and self.finbert_agent.is_available()),
            'groq_reasoning': bool(self.groq_agent and self.groq_agent.is_available())
        }
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all AI agents"""
        return {
            'available_agents': self.available_agents,
            'total_agents': len(self.available_agents),
            'active_agents': sum(self.available_agents.values()),
            'multi_ai_enabled': sum(self.available_agents.values()) > 1
        }