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
from .gemini_correlation_agent import GeminiCorrelationAgent
from ..logs.logger import get_logger

# Optional DeepSeek import with fallback
try:
    from .deepseek_agent import DeepSeekAgent
except Exception as e:
    DeepSeekAgent = None
    logger.warning(f"DeepSeek disabled: import failed: {e}")

logger = get_logger(__name__)

class MultiAIConsensus:
    """
    Advanced multi-AI consensus system that combines insights from:
    - Manus AI (Primary strategy recommendations)
    - Perplexity AI (Market intelligence)
    - Gemini AI (Cross-asset correlations)
    - DeepSeek AI (Advanced reasoning and sentiment analysis)
    """
    
    def __init__(self):
        # Initialize all AI agents
        self.manus_ai = ManusAI()
        self.perplexity_agent = PerplexityNewsAgent()
        self.gemini_agent = GeminiCorrelationAgent()
        self.deepseek_agent = DeepSeekAgent() if DeepSeekAgent else None
        
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
        
        if self.available_agents.get('gemini_correlation') and base_signal:
            tasks.append(self._run_gemini_analysis(symbol, base_signal.get('action', 'BUY')))
        
        if self.available_agents.get('deepseek_reasoning'):
            current_price = market_data['close'].iloc[-1] if len(market_data) > 0 else 0
            tasks.append(self._run_deepseek_analysis(symbol, market_data, current_price))
        
        # Execute all analyses concurrently
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"AI analysis task {i} failed: {result}")
                else:
                    analyses.update(result)
        
        # Generate consensus analysis
        consensus = self._generate_consensus(analyses, base_signal)
        
        logger.info(f"Multi-AI consensus for {symbol}: confidence={consensus.get('final_confidence', 0):.3f}, agents={len(analyses)}")
        
        return consensus
    
    async def _run_manus_analysis(self, symbol: str, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Run Manus AI analysis"""
        try:
            # Get Manus AI regime and strategy recommendations
            regime_data = self.manus_ai.detect_market_regime(market_data, symbol)
            strategy_recommendations = self.manus_ai.get_strategy_recommendations(symbol, regime_data)
            
            return {
                'manus_ai': {
                    'regime': regime_data.get('regime', 'UNKNOWN'),
                    'regime_confidence': regime_data.get('confidence', 0.5),
                    'recommended_strategies': strategy_recommendations.get('strategies', []),
                    'market_condition': strategy_recommendations.get('market_condition', 'neutral'),
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
    
    async def _run_gemini_analysis(self, symbol: str, signal_action: str) -> Dict[str, Any]:
        """Run Gemini correlation analysis"""
        try:
            analysis = self.gemini_agent.analyze_cross_asset_correlations(symbol, signal_action)
            return {'gemini_correlation': analysis}
        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            return {}
    
    async def _run_deepseek_analysis(self, symbol: str, market_data: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """Run DeepSeek sentiment and reasoning analysis"""
        try:
            analysis = await self.deepseek_agent.analyze_market_sentiment(symbol, market_data, current_price)
            return {'deepseek_reasoning': analysis}
        except Exception as e:
            logger.error(f"DeepSeek analysis failed: {e}")
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
                'weight': 0.3  # Primary weight for strategy recommendations
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
        
        # Process Gemini correlation insights
        if 'gemini_correlation' in analyses:
            gemini = analyses['gemini_correlation']
            correlation_impact = gemini.get('correlation_impact', 0.0)
            confidence_adjustments.append(('gemini_correlation', correlation_impact, 0.25))
            
            agent_insights['gemini_correlation'] = {
                'usd_strength': gemini.get('usd_strength', 'neutral'),
                'risk_sentiment': gemini.get('risk_sentiment', 'neutral'),
                'market_regime': gemini.get('market_regime', 'unknown'),
                'weight': 0.25
            }
        
        # Calculate weighted confidence adjustment
        total_adjustment = 0.0
        total_weight = 0.0
        
        for agent_name, adjustment, weight in confidence_adjustments:
            total_adjustment += adjustment * weight
            total_weight += weight
        
        # Apply confidence adjustment
        if total_weight > 0:
            weighted_adjustment = total_adjustment / total_weight
            final_confidence = base_confidence + weighted_adjustment
        else:
            final_confidence = base_confidence
        
        # Ensure confidence is within bounds
        final_confidence = max(0.0, min(1.0, final_confidence))
        
        # Generate consensus recommendation
        consensus_action = self._determine_consensus_action(agent_insights, base_signal)
        
        # Create detailed consensus analysis
        consensus = {
            'final_confidence': final_confidence,
            'base_confidence': base_confidence,
            'confidence_adjustment': final_confidence - base_confidence,
            'consensus_action': consensus_action,
            'agent_count': len(analyses),
            'agent_insights': agent_insights,
            'consensus_strength': self._calculate_consensus_strength(agent_insights),
            'risk_assessment': self._assess_overall_risk(agent_insights),
            'timestamp': datetime.now().isoformat(),
            'multi_ai_enabled': True
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
            if agent == 'perplexity_news':
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
            'gemini_correlation': self.gemini_agent.is_available(),
            'deepseek_reasoning': bool(self.deepseek_agent and self.deepseek_agent.is_available())
        }
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all AI agents"""
        return {
            'available_agents': self.available_agents,
            'total_agents': len(self.available_agents),
            'active_agents': sum(self.available_agents.values()),
            'multi_ai_enabled': sum(self.available_agents.values()) > 1
        }