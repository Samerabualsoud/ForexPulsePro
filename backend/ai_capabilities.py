"""
Enhanced AI Capabilities Detection Module
Centralized detection of available AI services and packages including multi-AI support
"""

import os
import warnings
from typing import Dict, Any, Optional

# Initialize logging first
from .logs.logger import get_logger
logger = get_logger(__name__)

# OpenAI capability flags
OPENAI_AVAILABLE = False
OPENAI_ENABLED = False
OPENAI_API_KEY = None

# Enhanced AI capability flags  
PERPLEXITY_AVAILABLE = False
PERPLEXITY_ENABLED = False
DEEPSEEK_AVAILABLE = False
DEEPSEEK_ENABLED = False
HUGGINGFACE_AVAILABLE = False
HUGGINGFACE_ENABLED = False

# Try to import OpenAI package
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
    logger.info("OpenAI package is available")
    
    # Check for API key
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    if OPENAI_API_KEY:
        # Validate API key by creating client (doesn't make API call)
        try:
            openai_client = OpenAI(api_key=OPENAI_API_KEY)
            OPENAI_ENABLED = True
            logger.info("OpenAI integration enabled with valid API key")
        except Exception as e:
            logger.warning(f"OpenAI API key validation failed: {e}")
            OPENAI_ENABLED = False
    else:
        logger.info("OpenAI package available but no API key provided (OPENAI_API_KEY)")
        OPENAI_ENABLED = False
        
except ImportError as e:
    logger.info(f"OpenAI package not available: {e}")
    OPENAI_AVAILABLE = False
    OPENAI_ENABLED = False

# Check Perplexity availability (uses requests, so just check API key)
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')
if PERPLEXITY_API_KEY:
    PERPLEXITY_AVAILABLE = True
    PERPLEXITY_ENABLED = True
    logger.info("Perplexity integration enabled")
else:
    logger.info("Perplexity API key not provided (PERPLEXITY_API_KEY)")


# Check DeepSeek availability (uses requests with OpenAI-compatible API)
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
if DEEPSEEK_API_KEY:
    try:
        import requests
        DEEPSEEK_AVAILABLE = True
        DEEPSEEK_ENABLED = True
        logger.info("DeepSeek integration enabled with API key")
    except ImportError:
        logger.warning("DeepSeek requires requests library")
        DEEPSEEK_AVAILABLE = False
        DEEPSEEK_ENABLED = False
else:
    logger.info("DeepSeek API key not provided (DEEPSEEK_API_KEY)")

# Check Hugging Face FinBERT availability (uses requests for inference API)
HUGGINGFACE_API_TOKEN = os.getenv('HUGGINGFACE_API_TOKEN')
if HUGGINGFACE_API_TOKEN:
    try:
        import requests
        HUGGINGFACE_AVAILABLE = True
        HUGGINGFACE_ENABLED = True
        logger.info("Hugging Face FinBERT integration enabled with API token")
    except ImportError:
        logger.warning("Hugging Face FinBERT requires requests library")
        HUGGINGFACE_AVAILABLE = False
        HUGGINGFACE_ENABLED = False
else:
    logger.info("Hugging Face API token not provided (HUGGINGFACE_API_TOKEN)")

# Calculate overall AI capabilities
MULTI_AI_ENABLED = sum([OPENAI_ENABLED, PERPLEXITY_ENABLED, DEEPSEEK_ENABLED, HUGGINGFACE_ENABLED]) > 1
TOTAL_AI_AGENTS = sum([True, PERPLEXITY_ENABLED, DEEPSEEK_ENABLED, HUGGINGFACE_ENABLED])  # Manus AI always available

logger.info(f"Multi-AI System Status: {TOTAL_AI_AGENTS} agents available, Multi-AI: {MULTI_AI_ENABLED}")

def get_ai_capabilities() -> Dict[str, Any]:
    """
    Get enhanced AI capabilities summary including multi-AI support
    
    Returns:
        Dict with capability flags and status information
    """
    return {
        'openai_available': OPENAI_AVAILABLE,
        'openai_enabled': OPENAI_ENABLED,
        'perplexity_available': PERPLEXITY_AVAILABLE,
        'perplexity_enabled': PERPLEXITY_ENABLED,
        'deepseek_available': DEEPSEEK_AVAILABLE,
        'deepseek_enabled': DEEPSEEK_ENABLED,
        'huggingface_available': HUGGINGFACE_AVAILABLE,
        'huggingface_enabled': HUGGINGFACE_ENABLED,
        'multi_ai_enabled': MULTI_AI_ENABLED,
        'total_ai_agents': TOTAL_AI_AGENTS,
        'dual_ai_mode': OPENAI_ENABLED,  # Legacy compatibility
        'manus_ai_only': not MULTI_AI_ENABLED,
        'capabilities': {
            'market_intelligence': PERPLEXITY_ENABLED,
            'deepseek_analysis': DEEPSEEK_ENABLED,
            'news_sentiment': HUGGINGFACE_ENABLED,
            'strategy_consensus': MULTI_AI_ENABLED,
            'advanced_backtesting': OPENAI_ENABLED,
            'chatgpt_optimization': OPENAI_ENABLED,
            'perplexity_news': PERPLEXITY_ENABLED,
            'deepseek_reasoning': DEEPSEEK_ENABLED,
            'finbert_sentiment': HUGGINGFACE_ENABLED,
            'fallback_manus_ai': True  # Always available
        }
    }

def create_openai_client() -> Optional[Any]:
    """
    Create OpenAI client if available and enabled
    
    Returns:
        OpenAI client instance or None if not available
    """
    if not OPENAI_ENABLED:
        return None
    
    try:
        from openai import OpenAI
        return OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        logger.error(f"Failed to create OpenAI client: {e}")
        return None

def create_deepseek_client() -> Optional[Any]:
    """
    Create DeepSeek client if available and enabled
    
    Returns:
        Simple requests-based client for DeepSeek API or None if not available
    """
    if not DEEPSEEK_ENABLED:
        return None
    
    try:
        import requests
        # Return a simple client config dict for DeepSeek API
        return {
            'api_key': DEEPSEEK_API_KEY,
            'base_url': 'https://api.deepseek.com/v1',
            'requests': requests
        }
    except Exception as e:
        logger.error(f"Failed to create DeepSeek client: {e}")
        return None

def create_finbert_client() -> Optional[Any]:
    """
    Create FinBERT client configuration if available and enabled
    
    Returns:
        Simple requests-based client config for Hugging Face API or None if not available
    """
    if not HUGGINGFACE_ENABLED:
        return None
    
    try:
        import requests
        # Return a simple client config dict for Hugging Face Inference API
        return {
            'api_token': HUGGINGFACE_API_TOKEN,
            'model_name': 'ProsusAI/finbert',
            'base_url': 'https://api-inference.huggingface.co/models/ProsusAI/finbert',
            'requests': requests
        }
    except Exception as e:
        logger.error(f"Failed to create FinBERT client: {e}")
        return None

def log_ai_status():
    """Log the current AI capability status"""
    capabilities = get_ai_capabilities()
    
    if capabilities['multi_ai_enabled']:
        active_agents = []
        if OPENAI_ENABLED: active_agents.append("ChatGPT")
        if PERPLEXITY_ENABLED: active_agents.append("Perplexity")
        if DEEPSEEK_ENABLED: active_agents.append("DeepSeek")
        if HUGGINGFACE_ENABLED: active_agents.append("FinBERT")
        
        logger.info(f"Multi-AI mode active: Manus AI + {', '.join(active_agents)}")
    elif capabilities['dual_ai_mode']:
        logger.info("Dual-AI mode active: Manus AI + ChatGPT consensus")
    else:
        if not OPENAI_AVAILABLE:
            logger.info("Single-AI mode: Manus AI only (OpenAI package not installed)")
        elif not OPENAI_ENABLED:
            logger.info("Single-AI mode: Manus AI only (OpenAI API key not configured)")
    
    logger.info(f"Available capabilities: {', '.join([k for k, v in capabilities['capabilities'].items() if v])}")

# Log status on module import
log_ai_status()