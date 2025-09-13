"""
Sentiment Factor Configuration

Central configuration for the sentiment analysis integration.
Environment variables can override these defaults.
"""

import os
from typing import Dict, Any


class SentimentConfig:
    """Configuration settings for sentiment factor analysis"""
    
    # Core settings
    ENABLED = os.getenv('SENTIMENT_FACTOR_ENABLED', 'true').lower() == 'true'
    WEIGHT = float(os.getenv('SENTIMENT_WEIGHT', '0.1'))  # 10% max impact on confidence
    LOOKBACK_HOURS = int(os.getenv('SENTIMENT_LOOKBACK_HOURS', '24'))  # 24 hours of news
    RECENCY_DECAY = float(os.getenv('SENTIMENT_RECENCY_DECAY', '0.5'))  # Exponential decay
    
    # Sentiment thresholds
    POSITIVE_THRESHOLD = float(os.getenv('SENTIMENT_POSITIVE_THRESHOLD', '0.3'))
    NEGATIVE_THRESHOLD = float(os.getenv('SENTIMENT_NEGATIVE_THRESHOLD', '-0.3'))
    HIGH_CONFIDENCE_THRESHOLD = float(os.getenv('SENTIMENT_HIGH_CONFIDENCE_THRESHOLD', '0.7'))
    
    # Performance settings
    MAX_ARTICLES_PER_SYMBOL = int(os.getenv('SENTIMENT_MAX_ARTICLES', '50'))
    MIN_ARTICLES_FOR_ANALYSIS = int(os.getenv('SENTIMENT_MIN_ARTICLES', '1'))
    
    # Fallback settings
    GRACEFUL_FALLBACK = os.getenv('SENTIMENT_GRACEFUL_FALLBACK', 'true').lower() == 'true'
    TIMEOUT_SECONDS = float(os.getenv('SENTIMENT_TIMEOUT', '5.0'))
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'enabled': cls.ENABLED,
            'weight': cls.WEIGHT,
            'lookback_hours': cls.LOOKBACK_HOURS,
            'recency_decay': cls.RECENCY_DECAY,
            'positive_threshold': cls.POSITIVE_THRESHOLD,
            'negative_threshold': cls.NEGATIVE_THRESHOLD,
            'high_confidence_threshold': cls.HIGH_CONFIDENCE_THRESHOLD,
            'max_articles_per_symbol': cls.MAX_ARTICLES_PER_SYMBOL,
            'min_articles_for_analysis': cls.MIN_ARTICLES_FOR_ANALYSIS,
            'graceful_fallback': cls.GRACEFUL_FALLBACK,
            'timeout_seconds': cls.TIMEOUT_SECONDS
        }
    
    @classmethod
    def get_environment_variables_doc(cls) -> str:
        """Get documentation for environment variables"""
        return """
# Sentiment Factor Environment Variables

# Enable/disable sentiment analysis integration
SENTIMENT_FACTOR_ENABLED=true              # Enable sentiment analysis (default: true)

# Core configuration
SENTIMENT_WEIGHT=0.1                       # Max impact on confidence (0.0-1.0, default: 0.1)
SENTIMENT_LOOKBACK_HOURS=24                # Hours of news to analyze (default: 24)
SENTIMENT_RECENCY_DECAY=0.5                # Time decay factor (default: 0.5)

# Sentiment thresholds
SENTIMENT_POSITIVE_THRESHOLD=0.3           # Positive sentiment threshold (default: 0.3)
SENTIMENT_NEGATIVE_THRESHOLD=-0.3          # Negative sentiment threshold (default: -0.3)
SENTIMENT_HIGH_CONFIDENCE_THRESHOLD=0.7    # High confidence threshold (default: 0.7)

# Performance tuning
SENTIMENT_MAX_ARTICLES=50                  # Max articles per symbol (default: 50)
SENTIMENT_MIN_ARTICLES=1                   # Min articles for analysis (default: 1)

# Reliability settings
SENTIMENT_GRACEFUL_FALLBACK=true           # Enable graceful fallback (default: true)
SENTIMENT_TIMEOUT=5.0                      # Analysis timeout in seconds (default: 5.0)

# Example usage in docker-compose.yml:
# environment:
#   - SENTIMENT_FACTOR_ENABLED=true
#   - SENTIMENT_WEIGHT=0.15
#   - SENTIMENT_LOOKBACK_HOURS=48
        """


# Global configuration instance
sentiment_config = SentimentConfig()