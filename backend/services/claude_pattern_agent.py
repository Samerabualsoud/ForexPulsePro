"""
Claude Pattern Agent - Advanced Technical Analysis and Chart Pattern Recognition
Using Anthropic Claude for deep market structure analysis
"""
import os
import json
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime

# Blueprint integration - Anthropic Claude
# The newest Anthropic model is "claude-sonnet-4-20250514", not "claude-3-7-sonnet-20250219", "claude-3-5-sonnet-20241022" nor "claude-3-sonnet-20240229". 
# Always prefer using "claude-sonnet-4-20250514" as it is the latest model.
try:
    import anthropic
    from anthropic import Anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False

from ..logs.logger import get_logger
logger = get_logger(__name__)

class ClaudePatternAgent:
    """Advanced chart pattern recognition and technical analysis using Claude AI"""
    
    def __init__(self):
        self.enabled = False
        self.client = None
        
        if CLAUDE_AVAILABLE:
            self.api_key = os.getenv('ANTHROPIC_API_KEY')
            if self.api_key:
                try:
                    self.client = Anthropic(api_key=self.api_key)
                    self.enabled = True
                    logger.info("Claude Pattern Agent initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize Claude client: {e}")
            else:
                logger.info("Claude Pattern Agent: ANTHROPIC_API_KEY not provided")
        else:
            logger.info("Claude Pattern Agent: Anthropic package not available")
    
    def analyze_chart_patterns(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Analyze chart patterns using Claude's advanced pattern recognition
        
        Args:
            data: OHLC price data
            symbol: Trading symbol
            
        Returns:
            Dict with pattern analysis including:
            - detected_patterns: List of identified chart patterns
            - confidence_adjustment: Signal confidence modifier (-0.3 to +0.3)
            - market_structure: Overall market structure assessment
            - key_levels: Important support/resistance levels
        """
        if not self.enabled:
            return self._fallback_analysis()
        
        try:
            # Prepare market data summary for Claude
            market_summary = self._prepare_market_summary(data, symbol)
            
            # Generate Claude analysis prompt
            prompt = self._create_analysis_prompt(market_summary, symbol)
            
            # Call Claude API
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=800,
                temperature=0.1,
                messages=[
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ]
            )
            
            # Parse Claude's response
            analysis = self._parse_claude_response(response.content[0].text)
            
            logger.info(f"Claude pattern analysis for {symbol}: {len(analysis.get('detected_patterns', []))} patterns detected")
            return analysis
            
        except Exception as e:
            logger.error(f"Claude pattern analysis failed for {symbol}: {e}")
            return self._fallback_analysis()
    
    def _prepare_market_summary(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Prepare concise market data summary for Claude analysis"""
        recent_data = data.tail(50)  # Last 50 bars
        
        current_price = recent_data['close'].iloc[-1]
        previous_price = recent_data['close'].iloc[-2] 
        price_change = ((current_price - previous_price) / previous_price) * 100
        
        # Calculate key technical levels
        high_20 = recent_data['high'].tail(20).max()
        low_20 = recent_data['low'].tail(20).min()
        sma_20 = recent_data['close'].tail(20).mean()
        
        # Recent price action
        recent_highs = recent_data['high'].tail(10).tolist()
        recent_lows = recent_data['low'].tail(10).tolist()
        recent_closes = recent_data['close'].tail(10).tolist()
        
        return {
            'symbol': symbol,
            'current_price': round(current_price, 5),
            'price_change_pct': round(price_change, 2),
            'high_20': round(high_20, 5),
            'low_20': round(low_20, 5),
            'sma_20': round(sma_20, 5),
            'recent_highs': [round(h, 5) for h in recent_highs],
            'recent_lows': [round(l, 5) for l in recent_lows],
            'recent_closes': [round(c, 5) for c in recent_closes],
            'volatility': round(recent_data['close'].pct_change().std() * 100, 2)
        }
    
    def _create_analysis_prompt(self, market_summary: Dict[str, Any], symbol: str) -> str:
        """Create analysis prompt for Claude"""
        return f"""
You are an expert forex technical analyst. Analyze the following market data for {symbol} and identify chart patterns and market structure.

Market Data:
- Symbol: {market_summary['symbol']}
- Current Price: {market_summary['current_price']}
- Price Change: {market_summary['price_change_pct']}%
- 20-period High: {market_summary['high_20']}
- 20-period Low: {market_summary['low_20']}
- 20-period SMA: {market_summary['sma_20']}
- Recent Volatility: {market_summary['volatility']}%

Recent Price Action (last 10 bars):
- Highs: {market_summary['recent_highs']}
- Lows: {market_summary['recent_lows']}
- Closes: {market_summary['recent_closes']}

Please analyze and provide:

1. CHART PATTERNS: Identify any classic chart patterns (head & shoulders, triangles, flags, pennants, double tops/bottoms, etc.)

2. MARKET STRUCTURE: Assess the overall trend and structure (uptrend, downtrend, ranging, consolidation)

3. KEY LEVELS: Identify important support and resistance levels

4. SIGNAL CONFIDENCE: Provide a confidence adjustment between -0.3 and +0.3 for trading signals based on pattern strength

5. TRADING BIAS: Overall bullish, bearish, or neutral bias

Respond in JSON format:
{
    "detected_patterns": ["pattern1", "pattern2"],
    "market_structure": "trend_description",
    "key_levels": {
        "support": [level1, level2],
        "resistance": [level1, level2]
    },
    "confidence_adjustment": 0.0,
    "trading_bias": "bullish/bearish/neutral",
    "reasoning": "brief_explanation"
}
"""
    
    def _parse_claude_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Claude's JSON response"""
        try:
            # Extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_text = response_text[start_idx:end_idx]
                analysis = json.loads(json_text)
                
                # Validate and sanitize response
                return {
                    'detected_patterns': analysis.get('detected_patterns', []),
                    'market_structure': analysis.get('market_structure', 'unknown'),
                    'key_levels': analysis.get('key_levels', {'support': [], 'resistance': []}),
                    'confidence_adjustment': max(-0.3, min(0.3, analysis.get('confidence_adjustment', 0.0))),
                    'trading_bias': analysis.get('trading_bias', 'neutral'),
                    'reasoning': analysis.get('reasoning', 'No reasoning provided'),
                    'agent': 'claude_pattern'
                }
            else:
                raise ValueError("No valid JSON found in response")
                
        except Exception as e:
            logger.error(f"Failed to parse Claude response: {e}")
            return self._fallback_analysis()
    
    def _fallback_analysis(self) -> Dict[str, Any]:
        """Fallback analysis when Claude is unavailable"""
        return {
            'detected_patterns': [],
            'market_structure': 'unknown',
            'key_levels': {'support': [], 'resistance': []},
            'confidence_adjustment': 0.0,
            'trading_bias': 'neutral',
            'reasoning': 'Claude Pattern Agent not available',
            'agent': 'claude_pattern_fallback'
        }
    
    def is_available(self) -> bool:
        """Check if Claude Pattern Agent is available"""
        return self.enabled