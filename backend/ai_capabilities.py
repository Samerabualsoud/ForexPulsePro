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
ANTHROPIC_AVAILABLE = False
ANTHROPIC_ENABLED = False
PERPLEXITY_AVAILABLE = False
PERPLEXITY_ENABLED = False
GEMINI_AVAILABLE = False
GEMINI_ENABLED = False

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

# Try to import and validate Anthropic (Claude)
try:
    import anthropic
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
    
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if api_key:
        try:
            anthropic_client = Anthropic(api_key=api_key)
            ANTHROPIC_ENABLED = True
            logger.info("Anthropic Claude integration enabled")
        except Exception as e:
            logger.warning(f"Anthropic API key validation failed: {e}")
    else:
        logger.info("Anthropic package available but no API key provided (ANTHROPIC_API_KEY)")
        
except ImportError as e:
    logger.info(f"Anthropic package not available: {e}")

# Check Perplexity availability (uses requests, so just check API key)
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')
if PERPLEXITY_API_KEY:
    PERPLEXITY_AVAILABLE = True
    PERPLEXITY_ENABLED = True
    logger.info("Perplexity integration enabled")
else:
    logger.info("Perplexity API key not provided (PERPLEXITY_API_KEY)")

# Try to import and validate Gemini
try:
    from google import genai
    GEMINI_AVAILABLE = True
    
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key:
        try:
            gemini_client = genai.Client(api_key=api_key)
            GEMINI_ENABLED = True
            logger.info("Google Gemini integration enabled")
        except Exception as e:
            logger.warning(f"Gemini API key validation failed: {e}")
    else:
        logger.info("Gemini package available but no API key provided (GEMINI_API_KEY)")
        
except ImportError as e:
    logger.info(f"Gemini package not available: {e}")

# Calculate overall AI capabilities
MULTI_AI_ENABLED = sum([OPENAI_ENABLED, ANTHROPIC_ENABLED, PERPLEXITY_ENABLED, GEMINI_ENABLED]) > 1
TOTAL_AI_AGENTS = sum([True, ANTHROPIC_ENABLED, PERPLEXITY_ENABLED, GEMINI_ENABLED])  # Manus AI always available

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
        'anthropic_available': ANTHROPIC_AVAILABLE,
        'anthropic_enabled': ANTHROPIC_ENABLED,
        'perplexity_available': PERPLEXITY_AVAILABLE,
        'perplexity_enabled': PERPLEXITY_ENABLED,
        'gemini_available': GEMINI_AVAILABLE,
        'gemini_enabled': GEMINI_ENABLED,
        'multi_ai_enabled': MULTI_AI_ENABLED,
        'total_ai_agents': TOTAL_AI_AGENTS,
        'dual_ai_mode': OPENAI_ENABLED,  # Legacy compatibility
        'manus_ai_only': not MULTI_AI_ENABLED,
        'capabilities': {
            'pattern_recognition': ANTHROPIC_ENABLED,
            'market_intelligence': PERPLEXITY_ENABLED,
            'correlation_analysis': GEMINI_ENABLED,
            'strategy_consensus': MULTI_AI_ENABLED,
            'advanced_backtesting': OPENAI_ENABLED or ANTHROPIC_ENABLED,
            'chatgpt_optimization': OPENAI_ENABLED,
            'claude_patterns': ANTHROPIC_ENABLED,
            'perplexity_news': PERPLEXITY_ENABLED,
            'gemini_correlations': GEMINI_ENABLED,
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

def log_ai_status():
    """Log the current AI capability status"""
    capabilities = get_ai_capabilities()
    
    if capabilities['dual_ai_mode']:
        logger.info("Dual-AI mode active: Manus AI + ChatGPT consensus")
    else:
        if not OPENAI_AVAILABLE:
            logger.info("Single-AI mode: Manus AI only (OpenAI package not installed)")
        elif not OPENAI_ENABLED:
            logger.info("Single-AI mode: Manus AI only (OpenAI API key not configured)")
    
    logger.info(f"Available capabilities: {', '.join([k for k, v in capabilities['capabilities'].items() if v])}")

# Log status on module import
log_ai_status()