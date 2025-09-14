"""
Resilience Utilities for AI Services
Provides robust error handling, retry logic, circuit breaker patterns, and rate limiting
"""
import asyncio
import time
import json
import random
from typing import Dict, Any, Optional, Callable, Union, List
from datetime import datetime, timedelta
from enum import Enum
import httpx
import requests
from dataclasses import dataclass, field

from ..logs.logger import get_logger

logger = get_logger(__name__)

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service has recovered

@dataclass
class RetryConfig:
    """Configuration for retry logic"""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    backoff_factor: float = 1.0

@dataclass  
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: tuple = (Exception,)
    
@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""
    requests_per_minute: int = 60
    burst_size: int = 10
    cooldown_seconds: float = 1.0

@dataclass
class CircuitBreakerState:
    """Circuit breaker state tracking"""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    state_changed_time: datetime = field(default_factory=datetime.now)

class RateLimiter:
    """Token bucket rate limiter for API requests"""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.tokens = config.burst_size
        self.last_refill = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """Acquire a token for making a request"""
        async with self._lock:
            now = time.time()
            # Refill tokens based on time elapsed
            elapsed = now - self.last_refill
            refill_amount = elapsed * (self.config.requests_per_minute / 60.0)
            self.tokens = min(self.config.burst_size, self.tokens + refill_amount)
            self.last_refill = now
            
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            return False
    
    async def wait_for_token(self) -> None:
        """Wait until a token is available"""
        while not await self.acquire():
            await asyncio.sleep(0.1)

class CircuitBreaker:
    """Circuit breaker for handling consecutive failures"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitBreakerState()
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function through the circuit breaker"""
        async with self._lock:
            if self.state.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state.state = CircuitState.HALF_OPEN
                    logger.info(f"Circuit breaker {self.name}: Attempting reset (HALF_OPEN)")
                else:
                    raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")
        
        try:
            # Use await-if-awaitable pattern to handle bound methods and partials
            import inspect
            result = func(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            await self._on_success()
            return result
        except self.config.expected_exception as e:
            await self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        if not self.state.last_failure_time:
            return True
        
        time_since_failure = datetime.now() - self.state.last_failure_time
        return time_since_failure.total_seconds() >= self.config.recovery_timeout
    
    async def _on_success(self) -> None:
        """Handle successful operation"""
        async with self._lock:
            self.state.failure_count = 0
            self.state.last_success_time = datetime.now()
            
            if self.state.state != CircuitState.CLOSED:
                self.state.state = CircuitState.CLOSED
                self.state.state_changed_time = datetime.now()
                logger.info(f"Circuit breaker {self.name}: Reset to CLOSED")
    
    async def _on_failure(self) -> None:
        """Handle failed operation"""
        async with self._lock:
            self.state.failure_count += 1
            self.state.last_failure_time = datetime.now()
            
            if (self.state.failure_count >= self.config.failure_threshold and 
                self.state.state == CircuitState.CLOSED):
                self.state.state = CircuitState.OPEN
                self.state.state_changed_time = datetime.now()
                logger.warning(f"Circuit breaker {self.name}: Opened after {self.state.failure_count} failures")
            elif self.state.state == CircuitState.HALF_OPEN:
                self.state.state = CircuitState.OPEN
                self.state.state_changed_time = datetime.now()
                logger.warning(f"Circuit breaker {self.name}: Back to OPEN from HALF_OPEN")

    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        return {
            'state': self.state.state.value,
            'failure_count': self.state.failure_count,
            'last_failure_time': self.state.last_failure_time.isoformat() if self.state.last_failure_time else None,
            'last_success_time': self.state.last_success_time.isoformat() if self.state.last_success_time else None,
            'state_changed_time': self.state.state_changed_time.isoformat()
        }

class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""
    pass

class ResilientAPIClient:
    """Resilient API client with retry, circuit breaker, and rate limiting"""
    
    def __init__(
        self, 
        name: str,
        retry_config: Optional[RetryConfig] = None,
        circuit_config: Optional[CircuitBreakerConfig] = None,
        rate_limit_config: Optional[RateLimitConfig] = None
    ):
        self.name = name
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breaker = CircuitBreaker(name, circuit_config) if circuit_config else None
        self.rate_limiter = RateLimiter(rate_limit_config) if rate_limit_config else None
        
        # HTTP clients with optimized settings
        self._http_client: Optional[httpx.AsyncClient] = None
        self._session_lock = asyncio.Lock()
        
        logger.info(f"ResilientAPIClient '{name}' initialized")
    
    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with optimized settings"""
        if self._http_client is None or self._http_client.is_closed:
            async with self._session_lock:
                if self._http_client is None or self._http_client.is_closed:
                    timeout = httpx.Timeout(
                        connect=10.0,
                        read=30.0,
                        write=10.0,
                        pool=60.0
                    )
                    
                    limits = httpx.Limits(
                        max_keepalive_connections=5,
                        max_connections=10,
                        keepalive_expiry=30.0
                    )
                    
                    self._http_client = httpx.AsyncClient(
                        timeout=timeout,
                        limits=limits,
                        headers={'User-Agent': f'ResilientAIClient/{self.name}'}
                    )
        
        return self._http_client
    
    async def make_request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, str]] = None,
        use_httpx: bool = True
    ) -> Union[httpx.Response, requests.Response]:
        """
        Make a resilient HTTP request with retry, circuit breaker, and rate limiting
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            headers: Request headers
            json_data: JSON payload
            data: Form data payload
            params: URL parameters
            use_httpx: Whether to use httpx (async) or requests (sync)
            
        Returns:
            HTTP response object
        """
        # Apply rate limiting if configured
        if self.rate_limiter:
            await self.rate_limiter.wait_for_token()
        
        # Define the async request function - always use httpx for consistency
        async def do_request():
            client = await self._get_http_client()
            # CRITICAL FIX: Properly await the httpx request
            response = await client.request(
                method=method,
                url=url,
                headers=headers,
                json=json_data,
                data=data,
                params=params
            )
            return response
        
        # Apply circuit breaker if configured, with null guard
        if self.circuit_breaker:
            async def circuit_wrapped_request():
                return await self.circuit_breaker.call(do_request)
            return await self._retry_with_backoff(circuit_wrapped_request)
        else:
            return await self._retry_with_backoff(do_request)
    
    async def _retry_with_backoff(self, func: Callable) -> Any:
        """Execute function with exponential backoff retry logic"""
        last_exception = None
        
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                # Always await since we're dealing with async functions
                result = await func()
                return result
                
            except Exception as e:
                last_exception = e
                
                if attempt == self.retry_config.max_retries:
                    logger.error(f"{self.name}: Final attempt failed after {attempt + 1} tries: {e}")
                    raise e
                
                # Calculate delay with exponential backoff
                delay = min(
                    self.retry_config.base_delay * (self.retry_config.exponential_base ** attempt) * self.retry_config.backoff_factor,
                    self.retry_config.max_delay
                )
                
                # Add jitter to prevent thundering herd
                if self.retry_config.jitter:
                    delay *= (0.5 + random.random() * 0.5)
                
                logger.warning(f"{self.name}: Attempt {attempt + 1} failed: {e}, retrying in {delay:.1f}s")
                await asyncio.sleep(delay)
        
        # Safe exception handling with fallback
        if last_exception:
            raise last_exception
        else:
            raise Exception(f"{self.name}: All retry attempts failed")
    
    async def close(self):
        """Close HTTP client connections"""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

class JSONParser:
    """Robust JSON parser with multiple fallback strategies"""
    
    @staticmethod
    def parse_json_response(content: str, agent_name: str = "unknown") -> Optional[Dict[str, Any]]:
        """
        Parse JSON with multiple fallback strategies
        
        Args:
            content: String content to parse
            agent_name: Name of the agent for logging
            
        Returns:
            Parsed JSON dict or None if all strategies fail
        """
        if not content or not content.strip():
            logger.warning(f"{agent_name}: Empty content provided for JSON parsing")
            return None
        
        # Strategy 1: Direct JSON parsing
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Extract JSON from markdown code blocks
        try:
            # Look for ```json blocks
            if '```json' in content:
                start = content.find('```json') + 7
                end = content.find('```', start)
                if end > start:
                    json_content = content[start:end].strip()
                    return json.loads(json_content)
        except json.JSONDecodeError:
            pass
        
        # Strategy 3: Extract first complete JSON object
        try:
            start = content.find('{')
            if start >= 0:
                brace_count = 0
                for i, char in enumerate(content[start:], start):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_content = content[start:i+1]
                            return json.loads(json_content)
        except json.JSONDecodeError:
            pass
        
        # Strategy 4: Extract using regex patterns
        try:
            import re
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(json_pattern, content)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        except Exception:
            pass
        
        # Strategy 5: Try to create a basic JSON structure from key-value patterns
        try:
            # Look for simple key: value patterns
            import re
            result = {}
            
            # Match "key": "value" or key: value patterns
            patterns = [
                r'"([^"]+)"\s*:\s*"([^"]*)"',  # "key": "value"
                r'"([^"]+)"\s*:\s*([0-9.]+)',   # "key": 123
                r'"([^"]+)"\s*:\s*(true|false|null)'  # "key": boolean/null
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for key, value in matches:
                    if value.lower() == 'true':
                        result[key] = True
                    elif value.lower() == 'false':
                        result[key] = False
                    elif value.lower() == 'null':
                        result[key] = None
                    elif value.replace('.', '').isdigit():
                        result[key] = float(value) if '.' in value else int(value)
                    else:
                        result[key] = value
            
            if result:
                logger.info(f"{agent_name}: Extracted partial JSON using pattern matching")
                return result
                
        except Exception as e:
            logger.debug(f"{agent_name}: Pattern extraction failed: {e}")
        
        logger.error(f"{agent_name}: All JSON parsing strategies failed for content: {content[:200]}...")
        return None

def create_ai_agent_client(
    agent_name: str,
    requests_per_minute: int = 60,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0
) -> ResilientAPIClient:
    """
    Create a pre-configured resilient API client for AI agents
    
    Args:
        agent_name: Name of the AI agent
        requests_per_minute: Rate limit for API requests
        failure_threshold: Number of failures before circuit opens
        recovery_timeout: Time to wait before attempting recovery
        
    Returns:
        Configured ResilientAPIClient instance
    """
    retry_config = RetryConfig(
        max_retries=3,
        base_delay=1.0,
        max_delay=30.0,
        exponential_base=2.0,
        jitter=True
    )
    
    circuit_config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        expected_exception=(Exception,)
    )
    
    rate_limit_config = RateLimitConfig(
        requests_per_minute=requests_per_minute,
        burst_size=min(10, requests_per_minute // 6),  # Allow burst of ~10 seconds worth
        cooldown_seconds=60.0 / requests_per_minute
    )
    
    return ResilientAPIClient(
        name=agent_name,
        retry_config=retry_config,
        circuit_config=circuit_config,
        rate_limit_config=rate_limit_config
    )

# Pre-configured clients for different AI services
def create_perplexity_client() -> ResilientAPIClient:
    """Create client optimized for Perplexity API (lower rate limits)"""
    return create_ai_agent_client(
        agent_name="Perplexity",
        requests_per_minute=20,  # Conservative rate limit for Perplexity
        failure_threshold=3,
        recovery_timeout=120.0
    )

def create_deepseek_client() -> ResilientAPIClient:
    """Create client optimized for DeepSeek API"""
    return create_ai_agent_client(
        agent_name="DeepSeek",
        requests_per_minute=30,
        failure_threshold=5,
        recovery_timeout=60.0
    )

def create_finbert_client() -> ResilientAPIClient:
    """Create client optimized for FinBERT/Hugging Face API"""
    return create_ai_agent_client(
        agent_name="FinBERT",
        requests_per_minute=60,
        failure_threshold=4,
        recovery_timeout=90.0
    )

def create_groq_client() -> ResilientAPIClient:
    """Create client optimized for Groq API"""
    return create_ai_agent_client(
        agent_name="Groq",
        requests_per_minute=60,
        failure_threshold=4,
        recovery_timeout=60.0
    )