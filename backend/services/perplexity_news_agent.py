"""
Perplexity News Agent - Real-Time Market Intelligence and News Analysis
Using Perplexity AI for live market context and economic event analysis
"""
import os
import json
import requests
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from ..logs.logger import get_logger
logger = get_logger(__name__)

class PerplexityNewsAgent:
    """Real-time market intelligence and news analysis using Perplexity AI"""
    
    def __init__(self):
        self.enabled = False
        self.api_key = os.getenv('PERPLEXITY_API_KEY')
        self.base_url = "https://api.perplexity.ai/chat/completions"
        
        if self.api_key:
            self.enabled = True
            logger.info("Perplexity News Agent initialized successfully")
        else:
            logger.info("Perplexity News Agent: PERPLEXITY_API_KEY not provided")
    
    def analyze_market_context(self, symbol: str, signal_action: str) -> Dict[str, Any]:
        """
        Analyze current market context and news events that might affect the signal
        
        Args:
            symbol: Trading symbol (e.g., EURUSD, BTCUSD)
            signal_action: BUY or SELL
            
        Returns:
            Dict with market context analysis including:
            - news_impact: Impact assessment (-0.2 to +0.2)
            - market_events: Relevant upcoming events
            - sentiment: Overall market sentiment
            - risk_factors: Potential risk factors
        """
        if not self.enabled:
            return self._fallback_analysis()
        
        try:
            # Create market context query
            query = self._create_market_query(symbol, signal_action)
            
            # Call Perplexity API
            response = self._call_perplexity_api(query)
            
            # Parse response
            analysis = self._parse_perplexity_response(response, symbol)
            
            logger.info(f"Perplexity market analysis for {symbol}: {analysis.get('sentiment', 'unknown')} sentiment")
            return analysis
            
        except Exception as e:
            logger.error(f"Perplexity market analysis failed for {symbol}: {e}")
            return self._fallback_analysis()
    
    def get_economic_calendar(self) -> Dict[str, Any]:
        """
        Get upcoming economic events that might impact forex markets
        
        Returns:
            Dict with upcoming economic events and their potential impact
        """
        if not self.enabled:
            return {'events': [], 'high_impact_count': 0}
        
        try:
            query = self._create_economic_calendar_query()
            response = self._call_perplexity_api(query)
            
            # Parse economic events
            events = self._parse_economic_events(response)
            
            logger.info(f"Perplexity economic calendar: {len(events.get('events', []))} events found")
            return events
            
        except Exception as e:
            logger.error(f"Perplexity economic calendar failed: {e}")
            return {'events': [], 'high_impact_count': 0}
    
    def _create_market_query(self, symbol: str, signal_action: str) -> str:
        """Create market context query for Perplexity"""
        base_currency, quote_currency = self._parse_symbol(symbol)
        
        return f"""
Analyze current market conditions and recent news for {symbol} ({base_currency} vs {quote_currency}). 
I'm considering a {signal_action} signal. Please provide:

1. Recent economic news affecting {base_currency} and {quote_currency}
2. Any central bank announcements or monetary policy changes
3. Geopolitical events impacting these currencies
4. Market sentiment and risk-on/risk-off environment
5. Any upcoming high-impact economic data releases today

Focus on events from the last 24 hours and upcoming events today. Be concise and factual.
"""
    
    def _create_economic_calendar_query(self) -> str:
        """Create economic calendar query"""
        today = datetime.now().strftime('%Y-%m-%d')
        return f"""
What are the most important economic events and data releases scheduled for today ({today}) and tomorrow that could impact forex markets? 

Focus on:
- Central bank speeches or announcements
- High-impact economic indicators (GDP, inflation, employment data)
- Policy decisions or rate changes
- Any breaking financial news

Provide the time (UTC if possible) and expected impact level (high/medium/low).
"""
    
    def _call_perplexity_api(self, query: str) -> Dict[str, Any]:
        """Call Perplexity API with the given query"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            "model": "llama-3.1-sonar-small-128k-online",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a financial markets expert providing factual, real-time market analysis. Be precise and concise."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            "max_tokens": 600,
            "temperature": 0.2,
            "top_p": 0.9,
            "search_recency_filter": "day",
            "return_images": False,
            "return_related_questions": False,
            "stream": False
        }
        
        response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        return response.json()
    
    def _parse_perplexity_response(self, response: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Parse Perplexity API response into structured analysis"""
        try:
            content = response['choices'][0]['message']['content']
            citations = response.get('citations', [])
            
            # Analyze content for market impact
            news_impact = self._assess_news_impact(content, symbol)
            sentiment = self._assess_sentiment(content)
            risk_factors = self._extract_risk_factors(content)
            
            return {
                'news_impact': news_impact,
                'sentiment': sentiment,
                'risk_factors': risk_factors,
                'analysis_text': content[:500],  # Truncate for storage
                'citations': citations[:3],  # Keep top 3 sources
                'agent': 'perplexity_news',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to parse Perplexity response: {e}")
            return self._fallback_analysis()
    
    def _parse_economic_events(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse economic calendar events from Perplexity response"""
        try:
            content = response['choices'][0]['message']['content']
            
            # Simple parsing of economic events
            # In a real implementation, you'd use more sophisticated NLP
            high_impact_keywords = ['rate decision', 'GDP', 'inflation', 'employment', 'central bank']
            high_impact_count = sum(1 for keyword in high_impact_keywords if keyword.lower() in content.lower())
            
            return {
                'events': [{'text': content[:300], 'impact': 'medium'}],  # Simplified
                'high_impact_count': min(high_impact_count, 3),
                'raw_analysis': content[:200]
            }
            
        except Exception as e:
            logger.error(f"Failed to parse economic events: {e}")
            return {'events': [], 'high_impact_count': 0}
    
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
    
    def _assess_news_impact(self, content: str, symbol: str) -> float:
        """Assess news impact on signal confidence"""
        content_lower = content.lower()
        
        # Positive impact keywords
        positive_keywords = ['bullish', 'positive', 'strong', 'growth', 'recovery', 'optimistic']
        negative_keywords = ['bearish', 'negative', 'weak', 'decline', 'recession', 'pessimistic']
        
        positive_score = sum(1 for word in positive_keywords if word in content_lower)
        negative_score = sum(1 for word in negative_keywords if word in content_lower)
        
        # Calculate impact (-0.2 to +0.2)
        impact = (positive_score - negative_score) * 0.05
        return max(-0.2, min(0.2, impact))
    
    def _assess_sentiment(self, content: str) -> str:
        """Assess overall market sentiment from news content"""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['bullish', 'positive', 'optimistic', 'growth']):
            return 'bullish'
        elif any(word in content_lower for word in ['bearish', 'negative', 'pessimistic', 'decline']):
            return 'bearish'
        else:
            return 'neutral'
    
    def _extract_risk_factors(self, content: str) -> List[str]:
        """Extract potential risk factors from news content"""
        risk_keywords = ['uncertainty', 'volatility', 'risk', 'concern', 'tension', 'crisis']
        content_lower = content.lower()
        
        found_risks = [keyword for keyword in risk_keywords if keyword in content_lower]
        return found_risks[:3]  # Limit to top 3
    
    def _fallback_analysis(self) -> Dict[str, Any]:
        """Fallback analysis when Perplexity is unavailable"""
        return {
            'news_impact': 0.0,
            'sentiment': 'neutral',
            'risk_factors': [],
            'analysis_text': 'Perplexity News Agent not available',
            'citations': [],
            'agent': 'perplexity_news_fallback',
            'timestamp': datetime.now().isoformat()
        }
    
    def is_available(self) -> bool:
        """Check if Perplexity News Agent is available"""
        return self.enabled