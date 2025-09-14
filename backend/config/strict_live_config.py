"""
Strict Live Mode Configuration

Enterprise-grade production safety configuration for live trading signals.
When STRICT_LIVE_MODE is enabled, ALL signals are blocked unless real market data requirements are met.
"""

import os
from typing import Dict, Any, List


class StrictLiveConfig:
    """Configuration settings for strict live mode - zero tolerance for non-real data"""
    
    # Core strict mode settings
    ENABLED = os.getenv('STRICT_LIVE_MODE', 'true').lower() == 'true'  # ENABLED FOR REAL DATA ONLY
    # For testing purposes - remove this line in production
    # ENABLED = True  # Uncomment this line to test strict mode functionality
    
    # Data freshness requirements (seconds)
    MAX_DATA_AGE_SECONDS = float(os.getenv('STRICT_LIVE_MAX_DATA_AGE', '15.0'))  # 15 seconds max
    
    # Provider validation requirements
    MIN_PROVIDER_VALIDATIONS = int(os.getenv('STRICT_LIVE_MIN_PROVIDERS', '1'))  # At least 1 provider must pass
    REQUIRE_LIVE_SOURCE = os.getenv('STRICT_LIVE_REQUIRE_LIVE_SOURCE', 'true').lower() == 'true'
    
    # Strict data requirements
    BLOCK_SYNTHETIC_DATA = os.getenv('STRICT_LIVE_BLOCK_SYNTHETIC', 'true').lower() == 'true'
    BLOCK_MOCK_DATA = os.getenv('STRICT_LIVE_BLOCK_MOCK', 'true').lower() == 'true'
    BLOCK_CACHED_DATA = os.getenv('STRICT_LIVE_BLOCK_CACHED', 'true').lower() == 'true'
    REQUIRE_REAL_DATA_MARKER = os.getenv('STRICT_LIVE_REQUIRE_REAL_MARKER', 'true').lower() == 'true'
    
    # Approved live data sources for strict mode
    APPROVED_LIVE_SOURCES = [
        source.strip() for source in os.getenv(
            'STRICT_LIVE_APPROVED_SOURCES',
            'Polygon.io,Finnhub,MT5,FreeCurrencyAPI,CoinGecko'
        ).split(',')
    ]
    
    # Sources that are NOT allowed in strict mode (cached/delayed data)
    BLOCKED_SOURCES = [
        source.strip() for source in os.getenv(
            'STRICT_LIVE_BLOCKED_SOURCES', 
            'ExchangeRate.host,AlphaVantage,MockDataProvider'
        ).split(',')
    ]
    
    # Market session validation
    REQUIRE_MARKET_OPEN = os.getenv('STRICT_LIVE_REQUIRE_MARKET_OPEN', 'true').lower() == 'true'
    
    # Safety thresholds
    MIN_DATA_BARS = int(os.getenv('STRICT_LIVE_MIN_DATA_BARS', '30'))  # Minimum bars for analysis
    
    # Logging settings
    VERBOSE_LOGGING = os.getenv('STRICT_LIVE_VERBOSE_LOGGING', 'true').lower() == 'true'
    LOG_ALL_VALIDATIONS = os.getenv('STRICT_LIVE_LOG_ALL_VALIDATIONS', 'true').lower() == 'true'
    
    @classmethod
    def is_data_source_approved(cls, source: str) -> bool:
        """Check if data source is approved for strict live mode"""
        if not cls.ENABLED:
            return True  # Allow all sources when strict mode is disabled
        
        return source in cls.APPROVED_LIVE_SOURCES and source not in cls.BLOCKED_SOURCES
    
    @classmethod
    def is_data_source_blocked(cls, source: str) -> bool:
        """Check if data source is explicitly blocked in strict mode"""
        if not cls.ENABLED:
            return False  # No sources blocked when strict mode is disabled
        
        return source in cls.BLOCKED_SOURCES
    
    @classmethod
    def validate_data_freshness(cls, data_age_seconds: float) -> tuple[bool, str]:
        """Validate data freshness against strict mode requirements"""
        if not cls.ENABLED:
            return True, "Strict mode disabled"
        
        if data_age_seconds <= cls.MAX_DATA_AGE_SECONDS:
            return True, f"Data age {data_age_seconds:.1f}s within limit ({cls.MAX_DATA_AGE_SECONDS}s)"
        else:
            return False, f"Data age {data_age_seconds:.1f}s exceeds strict limit ({cls.MAX_DATA_AGE_SECONDS}s)"
    
    @classmethod
    def get_status_summary(cls) -> Dict[str, Any]:
        """Get current strict mode status and configuration"""
        return {
            'strict_mode_enabled': cls.ENABLED,
            'max_data_age_seconds': cls.MAX_DATA_AGE_SECONDS,
            'min_provider_validations': cls.MIN_PROVIDER_VALIDATIONS,
            'require_live_source': cls.REQUIRE_LIVE_SOURCE,
            'block_synthetic_data': cls.BLOCK_SYNTHETIC_DATA,
            'block_mock_data': cls.BLOCK_MOCK_DATA,
            'block_cached_data': cls.BLOCK_CACHED_DATA,
            'require_real_data_marker': cls.REQUIRE_REAL_DATA_MARKER,
            'approved_live_sources': cls.APPROVED_LIVE_SOURCES,
            'blocked_sources': cls.BLOCKED_SOURCES,
            'require_market_open': cls.REQUIRE_MARKET_OPEN,
            'min_data_bars': cls.MIN_DATA_BARS,
            'verbose_logging': cls.VERBOSE_LOGGING
        }
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return cls.get_status_summary()
    
    @classmethod
    def get_environment_variables_doc(cls) -> str:
        """Get documentation for strict live mode environment variables"""
        return """
# Strict Live Mode Environment Variables
# Enterprise-grade production safety for live trading signals

# Core Settings
STRICT_LIVE_MODE=false                    # Enable strict live mode (default: false)
STRICT_LIVE_MAX_DATA_AGE=15.0            # Maximum data age in seconds (default: 15.0)
STRICT_LIVE_MIN_PROVIDERS=1              # Minimum providers that must pass validation (default: 1)
STRICT_LIVE_REQUIRE_LIVE_SOURCE=true     # Require live data source validation (default: true)

# Data Quality Requirements
STRICT_LIVE_BLOCK_SYNTHETIC=true         # Block synthetic/simulated data (default: true)
STRICT_LIVE_BLOCK_MOCK=true              # Block mock/test data (default: true)
STRICT_LIVE_BLOCK_CACHED=true            # Block cached/delayed data (default: true)
STRICT_LIVE_REQUIRE_REAL_MARKER=true     # Require explicit real data marker (default: true)

# Data Source Management
STRICT_LIVE_APPROVED_SOURCES="Polygon.io,Finnhub,MT5,FreeCurrencyAPI"    # Comma-separated approved sources
STRICT_LIVE_BLOCKED_SOURCES="ExchangeRate.host,AlphaVantage,MockDataProvider"  # Comma-separated blocked sources

# Safety Requirements
STRICT_LIVE_REQUIRE_MARKET_OPEN=true     # Require market to be open (default: true)
STRICT_LIVE_MIN_DATA_BARS=30             # Minimum data bars required (default: 30)

# Logging Configuration
STRICT_LIVE_VERBOSE_LOGGING=true         # Enable verbose strict mode logging (default: true)
STRICT_LIVE_LOG_ALL_VALIDATIONS=true     # Log all validation attempts (default: true)

# Example Production Configuration:
# export STRICT_LIVE_MODE=true
# export STRICT_LIVE_MAX_DATA_AGE=10.0
# export STRICT_LIVE_MIN_PROVIDERS=2
# export STRICT_LIVE_APPROVED_SOURCES="Polygon.io,MT5"
"""
    
    @classmethod
    def log_configuration(cls, logger) -> None:
        """Log current strict mode configuration"""
        if cls.ENABLED:
            logger.warning("🔒 STRICT LIVE MODE ENABLED - Production safety mode active")
            logger.info(f"   ├─ Max data age: {cls.MAX_DATA_AGE_SECONDS}s")
            logger.info(f"   ├─ Min providers: {cls.MIN_PROVIDER_VALIDATIONS}")
            logger.info(f"   ├─ Live source required: {cls.REQUIRE_LIVE_SOURCE}")
            logger.info(f"   ├─ Approved sources: {', '.join(cls.APPROVED_LIVE_SOURCES)}")
            logger.info(f"   ├─ Blocked sources: {', '.join(cls.BLOCKED_SOURCES)}")
            logger.info(f"   └─ Market open required: {cls.REQUIRE_MARKET_OPEN}")
        else:
            logger.info("🔓 Strict live mode DISABLED - Development mode active")