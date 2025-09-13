"""
DeepSeek AI Agent - Advanced Trading Analysis
Provides sophisticated market analysis and trading insights using DeepSeek AI
"""
import json
import asyncio
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime

from ..logs.logger import get_logger
from ..ai_capabilities import create_deepseek_client, DEEPSEEK_ENABLED

logger = get_logger(__name__)

class DeepSeekAgent:
    """
    DeepSeek AI Agent for advanced trading analysis and market insights
    """
    
    def __init__(self):
        self.client = create_deepseek_client()
        self.available = DEEPSEEK_ENABLED and self.client is not None
        
        if self.available:
            logger.info("DeepSeek AI agent initialized successfully")
        else:
            logger.warning("DeepSeek AI agent not available")
    
    async def analyze_market_sentiment(
        self, 
        symbol: str, 
        market_data: pd.DataFrame,
        current_price: float
    ) -> Dict[str, Any]:
        """
        Analyze market sentiment and provide trading insights using DeepSeek
        
        Args:
            symbol: Trading symbol
            market_data: OHLC price data
            current_price: Current market price
            
        Returns:
            Dict with sentiment analysis and trading insights
        """
        if not self.available:
            return {
                'available': False,
                'sentiment': 'neutral',
                'confidence': 0.0,
                'reasoning': 'DeepSeek agent not available'
            }
        
        try:
            # Prepare market data summary
            recent_data = market_data.tail(20)
            price_change = ((current_price - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]) * 100
            volatility = recent_data['close'].pct_change().std() * 100
            
            # Create analysis prompt
            prompt = f"""As an expert forex trader, analyze the following market data for {symbol}:

Current Price: ${current_price:.5f}
Price Change (20 periods): {price_change:.2f}%
Recent Volatility: {volatility:.2f}%

Recent Price Action:
{recent_data[['open', 'high', 'low', 'close']].to_string()}

Provide a JSON response with:
1. sentiment: "bullish", "bearish", or "neutral"
2. confidence: 0.0 to 1.0
3. key_factors: list of 3 main factors influencing your analysis
4. price_target: suggested price target for next 4 hours
5. risk_level: "low", "medium", or "high"
6. reasoning: brief explanation of your analysis

Focus on technical patterns, momentum, and market structure."""

            # Make API call to DeepSeek
            response = self._make_api_call(prompt)
            
            if response:
                return {
                    'available': True,
                    'agent': 'DeepSeek',
                    **response
                }
            else:
                raise Exception("No response from DeepSeek API")
                
        except Exception as e:
            logger.error(f"DeepSeek analysis failed for {symbol}: {e}")
            return {
                'available': False,
                'sentiment': 'neutral',
                'confidence': 0.0,
                'reasoning': f'Analysis failed: {str(e)}'
            }
    
    async def analyze_strategy_consensus(
        self, 
        symbol: str, 
        strategies: List[str],
        market_regime: str
    ) -> Dict[str, Any]:
        """
        Analyze strategy recommendations and provide consensus view
        
        Args:
            symbol: Trading symbol
            strategies: List of recommended strategies
            market_regime: Current market regime (TRENDING, RANGING, etc.)
            
        Returns:
            Dict with strategy analysis and recommendations
        """
        if not self.available:
            return {
                'available': False,
                'preferred_strategy': strategies[0] if strategies else 'ema_rsi',
                'confidence': 0.0
            }
        
        try:
            prompt = f"""As a quantitative trading expert, analyze this strategy selection for {symbol}:

Market Regime: {market_regime}
Recommended Strategies: {', '.join(strategies)}

Available strategies:
- ema_rsi: EMA crossover with RSI confirmation
- macd_crossover: MACD signal line crossovers
- meanrev_bb: Mean reversion using Bollinger Bands
- donchian_atr: Donchian channel breakouts with ATR sizing
- fibonacci: Fibonacci retracement levels
- rsi_divergence: RSI divergence patterns
- stochastic: Stochastic oscillator signals

Provide JSON response:
1. preferred_strategy: best strategy for current conditions
2. confidence: 0.0 to 1.0
3. reasoning: why this strategy is optimal
4. backup_strategy: second choice
5. avoid_strategies: strategies to avoid in current conditions"""

            response = self._make_api_call(prompt)
            
            if response:
                return {
                    'available': True,
                    'agent': 'DeepSeek',
                    **response
                }
            else:
                raise Exception("No response from DeepSeek API")
                
        except Exception as e:
            logger.error(f"DeepSeek strategy analysis failed for {symbol}: {e}")
            return {
                'available': False,
                'preferred_strategy': strategies[0] if strategies else 'ema_rsi',
                'confidence': 0.0,
                'reasoning': f'Analysis failed: {str(e)}'
            }
    
    def _make_api_call(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Make API call to DeepSeek with proper error handling
        
        Args:
            prompt: Analysis prompt
            
        Returns:
            Parsed JSON response or None if failed
        """
        if not self.client:
            return None
            
        try:
            requests = self.client['requests']
            
            headers = {
                'Authorization': f"Bearer {self.client['api_key']}",
                'Content-Type': 'application/json'
            }
            
            data = {
                'model': 'deepseek-reasoner',  # DeepSeek's reasoning model
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are an expert quantitative trader with deep knowledge of forex markets, technical analysis, and risk management. Always respond with valid JSON.'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                'temperature': 0.3,
                'max_tokens': 1000
            }
            
            response = requests.post(
                f"{self.client['base_url']}/chat/completions",
                headers=headers,
                json=data,
                timeout=10  # Reduced from 30s to 10s for better reliability
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # Try to parse JSON response
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # If not JSON, extract JSON from text
                    start = content.find('{')
                    end = content.rfind('}') + 1
                    if start >= 0 and end > start:
                        return json.loads(content[start:end])
                    else:
                        logger.warning("DeepSeek response not in JSON format")
                        return None
            else:
                logger.error(f"DeepSeek API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"DeepSeek API call failed: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if DeepSeek agent is available"""
        return self.available