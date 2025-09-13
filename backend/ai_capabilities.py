"""
AI Capabilities Detection Module
Centralized detection of available AI services and packages
"""

import os
import warnings
from typing import Dict, Any, Optional

# Initialize logging first
from .logs.logger import get_logger
logger = get_logger(__name__)

# Global capability flags
OPENAI_AVAILABLE = False
OPENAI_ENABLED = False
OPENAI_API_KEY = None

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

def get_ai_capabilities() -> Dict[str, Any]:
    """
    Get current AI capabilities summary
    
    Returns:
        Dict with capability flags and status information
    """
    return {
        'openai_available': OPENAI_AVAILABLE,
        'openai_enabled': OPENAI_ENABLED,
        'dual_ai_mode': OPENAI_ENABLED,  # Dual AI only works when OpenAI is fully enabled
        'manus_ai_only': not OPENAI_ENABLED,
        'capabilities': {
            'strategy_consensus': OPENAI_ENABLED,
            'advanced_backtesting': OPENAI_ENABLED,
            'chatgpt_optimization': OPENAI_ENABLED,
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