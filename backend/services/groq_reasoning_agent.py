"""
Groq Reasoning Agent - Advanced Market Analysis
Provides sophisticated market reasoning and analysis using Groq's fast inference models
"""
import json
import asyncio
import httpx
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime

from ..logs.logger import get_logger
from ..ai_capabilities import create_groq_client, GROQ_ENABLED

logger = get_logger(__name__)

class GroqReasoningAgent:
    """
    Groq AI Agent for advanced market reasoning and trading insights
    Specializes in fast inference and logical market analysis
    """
    
    def __init__(self):
        self.client = create_groq_client()
        self.available = GROQ_ENABLED and self.client is not None
        
        if self.available:
            logger.info("Groq Reasoning Agent initialized successfully")
        else:
            logger.warning("Groq Reasoning Agent not available")
    
    async def analyze_market_sentiment(
        self, 
        symbol: str, 
        market_data: pd.DataFrame,
        current_price: float
    ) -> Dict[str, Any]:
        """
        Analyze market sentiment and provide trading insights using Groq
        
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
                'reasoning': 'Groq agent not available'
            }
        
        try:
            # Input validation: Check for empty or insufficient data
            if market_data is None or market_data.empty:
                logger.warning(f"Empty market data provided for {symbol}")
                return {
                    'available': False,
                    'sentiment': 'neutral',
                    'confidence': 0.1,
                    'reasoning': 'Insufficient market data for analysis'
                }
            
            if len(market_data) < 5:
                logger.warning(f"Insufficient market data for {symbol}: only {len(market_data)} periods")
                return {
                    'available': False,
                    'sentiment': 'neutral', 
                    'confidence': 0.2,
                    'reasoning': f'Insufficient data: only {len(market_data)} periods available (minimum 5 required)'
                }
            
            # Prepare market data summary with safe data access
            recent_data = market_data.tail(20)
            
            # Safe price change calculation with fallback
            if len(recent_data) > 0 and 'close' in recent_data.columns:
                first_close = recent_data['close'].iloc[0]
                if first_close != 0:
                    price_change = ((current_price - first_close) / first_close) * 100
                else:
                    price_change = 0.0
            else:
                price_change = 0.0
                
            # Safe volatility calculation
            if len(recent_data) > 1 and 'close' in recent_data.columns:
                volatility = recent_data['close'].pct_change().std() * 100
                if pd.isna(volatility):
                    volatility = 0.0
            else:
                volatility = 0.0
            
            # Calculate additional market metrics with safety checks
            if len(recent_data) > 0 and all(col in recent_data.columns for col in ['high', 'low']):
                high_20 = recent_data['high'].max()
                low_20 = recent_data['low'].min()
                position_in_range = (current_price - low_20) / (high_20 - low_20) if high_20 != low_20 else 0.5
            else:
                high_20 = current_price
                low_20 = current_price
                position_in_range = 0.5
            
            # Create analysis prompt focused on logical reasoning
            prompt = f"""As an expert quantitative analyst, perform a logical market analysis for {symbol}:

MARKET DATA:
Current Price: ${current_price:.5f}
Price Change (20 periods): {price_change:.2f}%
Recent Volatility: {volatility:.2f}%
Position in 20-period range: {position_in_range:.1%}
Range High: ${high_20:.5f}
Range Low: ${low_20:.5f}

RECENT PRICE STRUCTURE:
{recent_data[['open', 'high', 'low', 'close']].tail(5).to_string()}

Provide a structured JSON response with:
1. sentiment: "bullish", "bearish", or "neutral"
2. confidence: 0.0 to 1.0 confidence in this assessment
3. market_structure: "trending_up", "trending_down", "ranging", or "reversal"
4. key_levels: array of 2-3 important price levels to watch
5. risk_factors: array of 2-3 main risks for this position
6. reasoning: detailed logical explanation of your analysis
7. time_horizon: "short" (1-4h), "medium" (4-12h), or "long" (12-24h)

Focus on logical price action analysis, support/resistance, and market structure."""

            # Make API call to Groq
            response = await self._make_api_call(prompt)
            
            if response:
                return {
                    'available': True,
                    'agent': 'Groq',
                    **response
                }
            else:
                raise Exception("No response from Groq API")
                
        except Exception as e:
            logger.error(f"Groq analysis failed for {symbol}: {e}")
            return {
                'available': False,
                'sentiment': 'neutral',
                'confidence': 0.0,
                'reasoning': f'Analysis failed: {str(e)}'
            }
    
    async def analyze_risk_assessment(
        self, 
        symbol: str, 
        signal_action: str,
        market_volatility: float
    ) -> Dict[str, Any]:
        """
        Analyze risk factors and provide risk assessment
        
        Args:
            symbol: Trading symbol
            signal_action: Proposed trading action (BUY/SELL)
            market_volatility: Current market volatility
            
        Returns:
            Dict with risk analysis and recommendations
        """
        if not self.available:
            return {
                'available': False,
                'risk_level': 'medium',
                'confidence': 0.0
            }
        
        try:
            prompt = f"""As a risk management expert, analyze this trading opportunity for {symbol}:

TRADE SETUP:
Proposed Action: {signal_action}
Market Volatility: {market_volatility:.2f}%
Symbol: {symbol}

Provide JSON response with:
1. risk_level: "low", "medium", or "high"
2. confidence: 0.0 to 1.0 confidence in risk assessment
3. risk_factors: array of specific risks identified
4. mitigation_strategies: array of risk mitigation suggestions
5. position_sizing: suggested position size (0.1 to 1.0 scale)
6. stop_loss_distance: suggested stop loss in percentage
7. reasoning: detailed risk analysis explanation

Consider market conditions, volatility, and currency-specific risks."""

            response = await self._make_api_call(prompt)
            
            if response:
                return {
                    'available': True,
                    'agent': 'Groq',
                    **response
                }
            else:
                raise Exception("No response from Groq API")
                
        except Exception as e:
            logger.error(f"Groq risk analysis failed for {symbol}: {e}")
            return {
                'available': False,
                'risk_level': 'medium',
                'confidence': 0.0,
                'reasoning': f'Risk analysis failed: {str(e)}'
            }
    
    async def _make_api_call(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Make async API call to Groq with proper error handling
        
        Args:
            prompt: Analysis prompt
            
        Returns:
            Parsed JSON response or None if failed
        """
        if not self.client:
            return None
            
        try:
            headers = {
                'Authorization': f"Bearer {self.client['api_key']}",
                'Content-Type': 'application/json'
            }
            
            data = {
                'model': 'llama-3.1-70b-versatile',  # Groq's fast inference model
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are an expert quantitative trader and risk analyst with deep knowledge of forex markets, technical analysis, and logical reasoning. Always respond with valid JSON and provide clear, logical explanations.'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                'temperature': 0.2,  # Lower temperature for more focused analysis
                'max_tokens': 1500,
                'top_p': 0.9
            }
            
            # Async retry logic for better reliability
            max_retries = 2
            for attempt in range(max_retries + 1):
                try:
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        response = await client.post(
                            f"{self.client['base_url']}/chat/completions",
                            headers=headers,
                            json=data
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
                                    logger.warning("Groq response not in JSON format")
                                    return None
                        else:
                            logger.error(f"Groq API error: {response.status_code} - {response.text}")
                            if attempt == max_retries:
                                return None
                            
                except (httpx.RequestError, httpx.HTTPStatusError) as e:
                    if attempt == max_retries:
                        logger.error(f"Groq API failed after {max_retries + 1} attempts: {e}")
                        return None
                    logger.warning(f"Groq API retry {attempt + 1}/{max_retries}: {e}")
                    await asyncio.sleep(0.5)  # Brief async delay before retry
                except Exception as e:
                    if attempt == max_retries:
                        logger.error(f"Groq API unexpected error after {max_retries + 1} attempts: {e}")
                        return None
                    logger.warning(f"Groq API unexpected error retry {attempt + 1}/{max_retries}: {e}")
                    await asyncio.sleep(0.5)
            
            return None
                
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if Groq agent is available"""
        return self.available