"""
FinBERT News Sentiment Agent - Financial News Analysis
Using Hugging Face ProsusAI/finbert model for sophisticated financial sentiment analysis
"""
import os
import json
import requests
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..logs.logger import get_logger
logger = get_logger(__name__)

class FinBERTSentimentAgent:
    """
    FinBERT agent for advanced financial news sentiment analysis using Hugging Face API
    Specializes in analyzing financial text and market news for trading signals
    """
    
    def __init__(self):
        self.enabled = False
        self.api_token = os.getenv('HUGGINGFACE_API_TOKEN')
        self.model_name = "ProsusAI/finbert"
        self.base_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        
        if self.api_token:
            self.enabled = True
            logger.info("FinBERT News Sentiment Agent initialized successfully")
        else:
            logger.info("FinBERT News Sentiment Agent: HUGGINGFACE_API_TOKEN not provided")
    
    async def analyze_news_sentiment(
        self, 
        symbol: str, 
        news_headlines: Optional[List[str]] = None,
        market_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze financial news sentiment for trading symbol using FinBERT
        
        Args:
            symbol: Trading symbol (e.g., EURUSD, BTCUSD)
            news_headlines: List of recent news headlines
            market_context: Optional market context information
            
        Returns:
            Dict with sentiment analysis including:
            - sentiment: Overall sentiment (bullish/bearish/neutral)
            - news_impact: Impact assessment (-0.2 to +0.2)
            - confidence: Confidence level (0.0 to 1.0)
            - risk_factors: Potential risk factors
            - reasoning: Analysis explanation
        """
        if not self.enabled:
            return self._fallback_analysis()
        
        try:
            # Create financial context for analysis
            financial_text = self._create_financial_context(symbol, news_headlines, market_context)
            
            # Analyze sentiment using FinBERT
            sentiment_result = await self._call_finbert_api(financial_text)
            
            if sentiment_result:
                # Parse FinBERT results into trading analysis
                analysis = self._parse_sentiment_analysis(sentiment_result, symbol)
                
                logger.info(f"FinBERT sentiment analysis for {symbol}: {analysis.get('sentiment', 'unknown')} (confidence: {analysis.get('confidence', 0):.3f})")
                return analysis
            else:
                raise Exception("No response from FinBERT API")
                
        except Exception as e:
            logger.error(f"FinBERT sentiment analysis failed for {symbol}: {e}")
            return self._fallback_analysis()
    
    async def analyze_market_news_impact(
        self, 
        symbol: str, 
        signal_action: str,
        recent_news: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze how recent financial news might impact a trading signal
        
        Args:
            symbol: Trading symbol
            signal_action: BUY or SELL signal
            recent_news: List of recent financial news items
            
        Returns:
            Dict with news impact analysis
        """
        if not self.enabled:
            return self._fallback_analysis()
        
        try:
            # Create context-aware analysis text
            market_text = self._create_market_impact_text(symbol, signal_action, recent_news)
            
            # Get sentiment analysis
            sentiment_result = await self._call_finbert_api(market_text)
            
            if sentiment_result:
                analysis = self._parse_market_impact(sentiment_result, symbol, signal_action)
                
                logger.info(f"FinBERT news impact for {symbol} {signal_action}: {analysis.get('news_impact', 0):.3f}")
                return analysis
            else:
                raise Exception("No response from FinBERT API")
                
        except Exception as e:
            logger.error(f"FinBERT market impact analysis failed for {symbol}: {e}")
            return self._fallback_analysis()
    
    async def _call_finbert_api(self, text: str) -> Optional[List[Dict[str, Any]]]:
        """
        Call Hugging Face Inference API for FinBERT sentiment analysis
        
        Args:
            text: Financial text to analyze
            
        Returns:
            FinBERT API response with sentiment scores
        """
        headers = {
            'Authorization': f'Bearer {self.api_token}',
            'Content-Type': 'application/json'
        }
        
        # FinBERT expects text input for classification
        payload = {
            "inputs": text[:512],  # Limit text length for FinBERT
            "options": {
                "wait_for_model": True,
                "use_cache": True
            }
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=15
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                return result  # Return the list directly for sentiment classification
            elif isinstance(result, dict):
                return [result]  # Wrap single dict in list for consistency
            else:
                logger.warning(f"Unexpected FinBERT response format: {type(result)}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"FinBERT API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"FinBERT API call failed: {e}")
            return None
    
    def _create_financial_context(
        self, 
        symbol: str, 
        news_headlines: Optional[List[str]], 
        market_context: Optional[str]
    ) -> str:
        """Create financial context text for FinBERT analysis"""
        base_currency, quote_currency = self._parse_symbol(symbol)
        
        context_parts = [
            f"Financial market analysis for {symbol} ({base_currency} vs {quote_currency})."
        ]
        
        if market_context:
            context_parts.append(f"Current market situation: {market_context}")
        
        if news_headlines:
            context_parts.append("Recent financial news:")
            context_parts.extend(news_headlines[:3])  # Limit to top 3 headlines
        else:
            context_parts.append(f"Analyzing general market sentiment for {base_currency} and {quote_currency} currencies.")
        
        return " ".join(context_parts)
    
    def _create_market_impact_text(
        self, 
        symbol: str, 
        signal_action: str, 
        recent_news: Optional[List[str]]
    ) -> str:
        """Create market impact analysis text"""
        base_currency, quote_currency = self._parse_symbol(symbol)
        
        text_parts = [
            f"Market impact analysis for {signal_action} signal on {symbol}.",
            f"Considering {base_currency} versus {quote_currency} trading opportunity."
        ]
        
        if recent_news:
            text_parts.append("Recent market developments:")
            text_parts.extend(recent_news[:2])  # Limit for context
        else:
            text_parts.append(f"Evaluating general market conditions affecting {base_currency} and {quote_currency}.")
        
        return " ".join(text_parts)
    
    def _parse_sentiment_analysis(
        self, 
        sentiment_result: List[Dict[str, Any]], 
        symbol: str
    ) -> Dict[str, Any]:
        """Parse FinBERT sentiment results into trading analysis"""
        try:
            # FinBERT returns sentiment classifications with scores
            # Expected labels: 'positive', 'negative', 'neutral'
            
            sentiments = {item['label'].lower(): item['score'] for item in sentiment_result}
            
            # Determine dominant sentiment
            max_sentiment = max(sentiments.keys(), key=lambda k: sentiments[k])
            max_confidence = sentiments[max_sentiment]
            
            # Map FinBERT sentiments to trading terms
            sentiment_mapping = {
                'positive': 'bullish',
                'negative': 'bearish', 
                'neutral': 'neutral'
            }
            
            trading_sentiment = sentiment_mapping.get(max_sentiment, 'neutral')
            
            # Calculate news impact (-0.2 to +0.2)
            news_impact = self._calculate_news_impact(sentiments)
            
            # Extract risk factors based on sentiment distribution
            risk_factors = self._extract_sentiment_risks(sentiments)
            
            return {
                'sentiment': trading_sentiment,
                'news_impact': news_impact,
                'confidence': max_confidence,
                'risk_factors': risk_factors,
                'reasoning': self._generate_reasoning(sentiments, trading_sentiment),
                'raw_scores': sentiments,
                'agent': 'finbert_sentiment',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to parse FinBERT sentiment results: {e}")
            return self._fallback_analysis()
    
    def _parse_market_impact(
        self, 
        sentiment_result: List[Dict[str, Any]], 
        symbol: str, 
        signal_action: str
    ) -> Dict[str, Any]:
        """Parse FinBERT results specifically for market impact assessment"""
        try:
            analysis = self._parse_sentiment_analysis(sentiment_result, symbol)
            
            # Adjust impact based on signal alignment
            sentiment = analysis.get('sentiment', 'neutral')
            base_impact = analysis.get('news_impact', 0.0)
            
            # Enhance impact if sentiment aligns with signal
            if (signal_action == 'BUY' and sentiment == 'bullish') or \
               (signal_action == 'SELL' and sentiment == 'bearish'):
                adjusted_impact = min(0.2, base_impact * 1.2)  # Boost aligned signals
            elif (signal_action == 'BUY' and sentiment == 'bearish') or \
                 (signal_action == 'SELL' and sentiment == 'bullish'):
                adjusted_impact = max(-0.2, base_impact * 1.2)  # Penalize conflicting signals
            else:
                adjusted_impact = base_impact
            
            analysis['news_impact'] = adjusted_impact
            analysis['signal_alignment'] = self._assess_signal_alignment(sentiment, signal_action)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to parse market impact: {e}")
            return self._fallback_analysis()
    
    def _calculate_news_impact(self, sentiments: Dict[str, float]) -> float:
        """Calculate news impact score from sentiment distribution"""
        positive_score = sentiments.get('positive', 0.0)
        negative_score = sentiments.get('negative', 0.0)
        neutral_score = sentiments.get('neutral', 0.0)
        
        # Calculate impact based on sentiment strength and distribution
        if positive_score > 0.6:
            impact = 0.15 * positive_score  # Strong positive
        elif negative_score > 0.6:
            impact = -0.15 * negative_score  # Strong negative
        elif positive_score > negative_score:
            impact = 0.1 * (positive_score - negative_score)
        elif negative_score > positive_score:
            impact = -0.1 * (negative_score - positive_score)
        else:
            impact = 0.0  # Neutral or mixed
        
        # Ensure within bounds
        return max(-0.2, min(0.2, impact))
    
    def _extract_sentiment_risks(self, sentiments: Dict[str, float]) -> List[str]:
        """Extract risk factors based on sentiment analysis"""
        risks = []
        
        negative_score = sentiments.get('negative', 0.0)
        neutral_score = sentiments.get('neutral', 0.0)
        
        if negative_score > 0.5:
            risks.append('negative_sentiment')
        
        if neutral_score > 0.4 and max(sentiments.values()) < 0.6:
            risks.append('sentiment_uncertainty')
        
        # Check for mixed sentiments (high uncertainty)
        sentiment_values = list(sentiments.values())
        if len(sentiment_values) >= 2 and max(sentiment_values) - min(sentiment_values) < 0.3:
            risks.append('mixed_signals')
        
        return risks[:3]  # Limit to top 3 risk factors
    
    def _generate_reasoning(self, sentiments: Dict[str, float], trading_sentiment: str) -> str:
        """Generate reasoning explanation for the sentiment analysis"""
        max_score = max(sentiments.values())
        
        if trading_sentiment == 'bullish':
            return f"Financial news shows positive sentiment (confidence: {max_score:.2f}), indicating potential upward market movement."
        elif trading_sentiment == 'bearish':
            return f"Financial news shows negative sentiment (confidence: {max_score:.2f}), suggesting potential downward market pressure."
        else:
            return f"Financial news shows neutral sentiment (confidence: {max_score:.2f}), indicating balanced market conditions."
    
    def _assess_signal_alignment(self, sentiment: str, signal_action: str) -> str:
        """Assess how well sentiment aligns with trading signal"""
        if (signal_action == 'BUY' and sentiment == 'bullish') or \
           (signal_action == 'SELL' and sentiment == 'bearish'):
            return 'aligned'
        elif (signal_action == 'BUY' and sentiment == 'bearish') or \
             (signal_action == 'SELL' and sentiment == 'bullish'):
            return 'conflicting'
        else:
            return 'neutral'
    
    def _parse_symbol(self, symbol: str) -> tuple:
        """Parse trading symbol into base and quote currencies"""
        if len(symbol) == 6:  # Standard forex pair like EURUSD
            return symbol[:3], symbol[3:]
        elif 'USD' in symbol:  # Crypto pairs like BTCUSD
            if symbol.endswith('USD'):
                return symbol[:-3], 'USD'
            else:
                return 'USD', symbol[3:]
        else:
            return symbol, 'USD'  # Default
    
    def _fallback_analysis(self) -> Dict[str, Any]:
        """Fallback analysis when FinBERT is unavailable"""
        return {
            'sentiment': 'neutral',
            'news_impact': 0.0,
            'confidence': 0.0,
            'risk_factors': ['finbert_unavailable'],
            'reasoning': 'FinBERT sentiment agent not available',
            'agent': 'finbert_sentiment_fallback',
            'timestamp': datetime.now().isoformat()
        }
    
    def is_available(self) -> bool:
        """Check if FinBERT sentiment agent is available"""
        return self.enabled