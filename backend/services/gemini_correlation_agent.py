"""
Gemini Correlation Agent - Cross-Asset Analysis and Market Correlation
Using Google Gemini for multi-modal market analysis and asset correlation insights
"""
import os
import json
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime

# Blueprint integration - Google Gemini
# The newest Gemini model series is "gemini-2.5-flash" or "gemini-2.5-pro"
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from ..logs.logger import get_logger
logger = get_logger(__name__)

class GeminiCorrelationAgent:
    """Cross-asset analysis and market correlation insights using Gemini AI"""
    
    def __init__(self):
        self.enabled = False
        self.client = None
        
        if GEMINI_AVAILABLE:
            self.api_key = os.getenv('GEMINI_API_KEY')
            if self.api_key:
                try:
                    self.client = genai.Client(api_key=self.api_key)
                    self.enabled = True
                    logger.info("Gemini Correlation Agent initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize Gemini client: {e}")
            else:
                logger.info("Gemini Correlation Agent: GEMINI_API_KEY not provided")
        else:
            logger.info("Gemini Correlation Agent: Google GenAI package not available")
    
    def analyze_cross_asset_correlations(self, symbol: str, signal_action: str) -> Dict[str, Any]:
        """
        Analyze cross-asset correlations and their impact on the trading signal
        
        Args:
            symbol: Trading symbol
            signal_action: BUY or SELL
            
        Returns:
            Dict with correlation analysis including:
            - usd_strength: USD strength assessment
            - risk_sentiment: Risk-on/risk-off sentiment
            - correlation_impact: Signal confidence adjustment
            - related_assets: Analysis of related asset movements
        """
        if not self.enabled:
            return self._fallback_analysis()
        
        try:
            # Create correlation analysis prompt
            prompt = self._create_correlation_prompt(symbol, signal_action)
            
            # Call Gemini API
            response = self.client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[
                    types.Content(role="user", parts=[types.Part(text=prompt)])
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                    max_output_tokens=600
                )
            )
            
            # Parse response
            analysis = self._parse_gemini_response(response.text, symbol)
            
            logger.info(f"Gemini correlation analysis for {symbol}: {analysis.get('risk_sentiment', 'unknown')} sentiment")
            return analysis
            
        except Exception as e:
            logger.error(f"Gemini correlation analysis failed for {symbol}: {e}")
            return self._fallback_analysis()
    
    def analyze_market_regime(self, current_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze current market regime and its implications for trading
        
        Args:
            current_conditions: Current market conditions data
            
        Returns:
            Dict with market regime analysis
        """
        if not self.enabled:
            return {'regime': 'unknown', 'confidence': 0.5}
        
        try:
            prompt = self._create_regime_prompt(current_conditions)
            
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )
            
            regime_analysis = self._parse_regime_response(response.text)
            
            logger.info(f"Gemini market regime: {regime_analysis.get('regime', 'unknown')}")
            return regime_analysis
            
        except Exception as e:
            logger.error(f"Gemini regime analysis failed: {e}")
            return {'regime': 'unknown', 'confidence': 0.5}
    
    def _create_correlation_prompt(self, symbol: str, signal_action: str) -> str:
        """Create correlation analysis prompt for Gemini"""
        market_type = self._identify_market_type(symbol)
        
        return f"""
Analyze cross-asset correlations and market dynamics for a {signal_action} signal on {symbol}.

Symbol Details:
- Asset: {symbol}
- Market Type: {market_type}
- Signal Direction: {signal_action}

Please analyze the following correlations and provide insights:

1. USD STRENGTH: If this is a USD pair, assess current USD strength/weakness
2. RISK SENTIMENT: Current risk-on vs risk-off environment
3. ASSET CORRELATIONS: How related assets (bonds, commodities, other currencies) might affect this trade
4. MARKET REGIME: Current market regime (trending, ranging, volatile, calm)

For cryptocurrency pairs, consider:
- Bitcoin correlation with traditional markets
- Crypto market sentiment and institutional flows

For forex pairs, consider:
- Central bank policy divergence
- Economic data differentials
- Safe haven flows

For metals/commodities, consider:
- Inflation expectations
- USD strength impact
- Industrial demand factors

Provide analysis in JSON format:
{{
    "usd_strength": "strong/weak/neutral",
    "risk_sentiment": "risk_on/risk_off/neutral", 
    "correlation_impact": 0.0,
    "related_assets": ["asset1", "asset2"],
    "market_regime": "trending/ranging/volatile",
    "confidence_adjustment": 0.0,
    "reasoning": "brief_explanation"
}}

Where correlation_impact and confidence_adjustment are between -0.2 and +0.2.
"""
    
    def _create_regime_prompt(self, conditions: Dict[str, Any]) -> str:
        """Create market regime analysis prompt"""
        return f"""
Analyze the current market regime based on the following conditions:

Market Conditions:
{json.dumps(conditions, indent=2)}

Please identify the current market regime and provide confidence level.

Possible regimes:
- BULL_MARKET: Strong upward trends across markets
- BEAR_MARKET: Broad market decline
- RISK_ON: Growth-focused, risk-seeking environment  
- RISK_OFF: Safety-focused, flight to quality
- VOLATILE: High uncertainty, choppy markets
- CALM: Low volatility, stable conditions
- TRANSITIONAL: Regime change in progress

Respond in JSON format:
{{
    "regime": "regime_name",
    "confidence": 0.0,
    "characteristics": ["characteristic1", "characteristic2"],
    "implications": "trading_implications"
}}

Where confidence is between 0 and 1.
"""
    
    def _parse_gemini_response(self, response_text: str, symbol: str) -> Dict[str, Any]:
        """Parse Gemini correlation response"""
        try:
            analysis = json.loads(response_text)
            
            # Validate and sanitize response
            return {
                'usd_strength': analysis.get('usd_strength', 'neutral'),
                'risk_sentiment': analysis.get('risk_sentiment', 'neutral'),
                'correlation_impact': max(-0.2, min(0.2, analysis.get('correlation_impact', 0.0))),
                'related_assets': analysis.get('related_assets', [])[:5],  # Limit to 5
                'market_regime': analysis.get('market_regime', 'unknown'),
                'confidence_adjustment': max(-0.2, min(0.2, analysis.get('confidence_adjustment', 0.0))),
                'reasoning': analysis.get('reasoning', 'No reasoning provided'),
                'agent': 'gemini_correlation'
            }
            
        except Exception as e:
            logger.error(f"Failed to parse Gemini response: {e}")
            return self._fallback_analysis()
    
    def _parse_regime_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Gemini regime response"""
        try:
            analysis = json.loads(response_text)
            
            return {
                'regime': analysis.get('regime', 'unknown'),
                'confidence': max(0.0, min(1.0, analysis.get('confidence', 0.5))),
                'characteristics': analysis.get('characteristics', [])[:3],
                'implications': analysis.get('implications', 'No implications provided')
            }
            
        except Exception as e:
            logger.error(f"Failed to parse Gemini regime response: {e}")
            return {'regime': 'unknown', 'confidence': 0.5}
    
    def _identify_market_type(self, symbol: str) -> str:
        """Identify the type of market for the symbol"""
        if symbol in ['BTCUSD', 'ETHUSD', 'BTCEUR', 'ETHEUR']:
            return 'cryptocurrency'
        elif symbol in ['XAUUSD', 'XAGUSD', 'USOIL', 'UKOUSD']:
            return 'commodity'
        else:
            return 'forex'
    
    def _fallback_analysis(self) -> Dict[str, Any]:
        """Fallback analysis when Gemini is unavailable"""
        return {
            'usd_strength': 'neutral',
            'risk_sentiment': 'neutral',
            'correlation_impact': 0.0,
            'related_assets': [],
            'market_regime': 'unknown',
            'confidence_adjustment': 0.0,
            'reasoning': 'Gemini Correlation Agent not available',
            'agent': 'gemini_correlation_fallback'
        }
    
    def is_available(self) -> bool:
        """Check if Gemini Correlation Agent is available"""
        return self.enabled