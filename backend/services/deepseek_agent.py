"""
DeepSeek AI Agent - Advanced Trading Analysis
Provides sophisticated market analysis and trading insights using DeepSeek AI
"""
import json
import asyncio
import random
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime, timedelta
import httpx

from ..logs.logger import get_logger
from ..ai_capabilities import create_deepseek_client, DEEPSEEK_ENABLED

logger = get_logger(__name__)

class DeepSeekAgent:
    """
    DeepSeek AI Agent for advanced trading analysis and market insights
    """
    
    def __init__(self):
        self.client_config = create_deepseek_client()
        self.available = DEEPSEEK_ENABLED and self.client_config is not None
        
        # Health tracking for fail-fast logic
        self.consecutive_failures = 0
        self.last_failure_time = None
        self.health_check_interval = 300  # 5 minutes
        self.max_consecutive_failures = 3
        
        # HTTP client configuration
        self.http_client = None
        self._session_lock = asyncio.Lock()
        
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

            # Make async API call to DeepSeek
            response = await self._make_api_call_async(prompt)
            
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

            response = await self._make_api_call_async(prompt)
            
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
    
    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with connection pooling"""
        if self.http_client is None or self.http_client.is_closed:
            async with self._session_lock:
                if self.http_client is None or self.http_client.is_closed:
                    # Create new client with optimized settings
                    timeout = httpx.Timeout(
                        connect=10.0,    # Connection timeout: 10s
                        read=90.0,       # Read timeout: 90s for AI reasoning models  
                        write=10.0,      # Write timeout: 10s
                        pool=120.0       # Overall request timeout: 120s
                    )
                    
                    limits = httpx.Limits(
                        max_keepalive_connections=5,
                        max_connections=10,
                        keepalive_expiry=30.0
                    )
                    
                    self.http_client = httpx.AsyncClient(
                        timeout=timeout,
                        limits=limits,
                        follow_redirects=True,
                        verify=True
                    )
        
        return self.http_client

    def _is_health_check_needed(self) -> bool:
        """Check if we should temporarily disable due to consecutive failures"""
        if self.consecutive_failures < self.max_consecutive_failures:
            return False
            
        if self.last_failure_time is None:
            return False
            
        time_since_failure = datetime.now() - self.last_failure_time
        return time_since_failure.total_seconds() < self.health_check_interval

    def _record_success(self):
        """Record successful API call"""
        self.consecutive_failures = 0
        self.last_failure_time = None

    def _record_failure(self):
        """Record failed API call"""
        self.consecutive_failures += 1
        self.last_failure_time = datetime.now()
        
        if self.consecutive_failures >= self.max_consecutive_failures:
            logger.warning(f"DeepSeek API marked as unhealthy after {self.consecutive_failures} consecutive failures")

    async def _make_api_call_async(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Make async API call to DeepSeek with proper error handling and fail-fast logic
        
        Args:
            prompt: Analysis prompt
            
        Returns:
            Parsed JSON response or None if failed
        """
        if not self.client_config:
            return None

        # Check health before making request
        if self._is_health_check_needed():
            logger.warning(f"DeepSeek API temporarily disabled due to {self.consecutive_failures} consecutive failures")
            return None
            
        try:
            client = await self._get_http_client()
            
            headers = {
                'Authorization': f"Bearer {self.client_config['api_key']}",
                'Content-Type': 'application/json',
                'User-Agent': 'ForexSignalDashboard/1.0'
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
            
            # Improved retry logic with exponential backoff and jitter
            max_retries = 3  # Now matches comment: 2s, 4s, 8s
            response = None
            
            for attempt in range(max_retries + 1):
                try:
                    response = await client.post(
                        f"{self.client_config['base_url']}/chat/completions",
                        headers=headers,
                        json=data
                    )
                    
                    # Don't record success yet - validate response first
                    break
                    
                except (httpx.TimeoutException, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.PoolTimeout) as e:
                    error_type = type(e).__name__
                    if attempt == max_retries:
                        logger.error(f"DeepSeek API {error_type} after {max_retries + 1} attempts: {e}")
                        self._record_failure()
                        return None
                    
                    # Exponential backoff with jitter: 2-5s, 4-8s, 8-16s
                    base_delay = 2 ** (attempt + 1)
                    jitter = random.uniform(0.0, base_delay)
                    delay = base_delay + jitter
                    logger.warning(f"DeepSeek API {error_type} retry {attempt + 1}/{max_retries}, waiting {delay:.1f}s: {e}")
                    await asyncio.sleep(delay)
                    
                except (httpx.ConnectError, httpx.RequestError, httpx.HTTPStatusError) as e:
                    logger.error(f"DeepSeek API network/request error: {e}")
                    self._record_failure()
                    return None
                    
                except Exception as e:
                    if attempt == max_retries:
                        logger.error(f"DeepSeek API unexpected error after {max_retries + 1} attempts: {e}")
                        self._record_failure()
                        return None
                    
                    # Exponential backoff with jitter for unexpected errors
                    base_delay = 2 ** (attempt + 1)
                    jitter = random.uniform(0.0, base_delay)
                    delay = base_delay + jitter
                    logger.warning(f"DeepSeek API retry {attempt + 1}/{max_retries} for error: {e}, waiting {delay:.1f}s")
                    await asyncio.sleep(delay)
            
            if response is None:
                self._record_failure()
                return None
            
            # Process response
            if response.status_code == 200:
                try:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    
                    # Try to parse JSON response
                    try:
                        parsed_response = json.loads(content)
                        # Success - record after validation
                        self._record_success()
                        return parsed_response
                    except json.JSONDecodeError:
                        # If not JSON, extract JSON from text
                        start = content.find('{')
                        end = content.rfind('}') + 1
                        if start >= 0 and end > start:
                            try:
                                parsed_response = json.loads(content[start:end])
                                # Success - record after validation
                                self._record_success()
                                return parsed_response
                            except json.JSONDecodeError:
                                logger.warning("DeepSeek response JSON extraction failed")
                                self._record_failure()
                                return None
                        else:
                            logger.warning("DeepSeek response not in JSON format")
                            self._record_failure()
                            return None
                            
                except Exception as e:
                    logger.error(f"DeepSeek response parsing failed: {e}")
                    self._record_failure()
                    return None
                    
            elif response.status_code == 429:
                logger.warning(f"DeepSeek API rate limited: {response.status_code}")
                self._record_failure()
                return None
            elif response.status_code >= 500:
                logger.error(f"DeepSeek API server error: {response.status_code} - {response.text}")
                self._record_failure()
                return None
            else:
                logger.error(f"DeepSeek API client error: {response.status_code} - {response.text}")
                self._record_failure()
                return None
                
        except Exception as e:
            logger.error(f"DeepSeek API call failed: {e}")
            self._record_failure()
            return None

    async def cleanup(self):
        """Cleanup HTTP client resources"""
        if self.http_client and not self.http_client.is_closed:
            await self.http_client.aclose()
    
    def is_available(self) -> bool:
        """Check if DeepSeek agent is available"""
        return self.available