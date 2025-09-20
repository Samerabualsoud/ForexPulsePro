"""
Deterministic Provider Configuration System
Ensures identical provider routing behavior between development and production environments
"""

import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from ..logs.logger import get_logger

logger = get_logger(__name__)

class ProviderType(Enum):
    """Provider classification for routing decisions"""
    LIVE_REAL_TIME = "live_real_time"      # Live, real-time data (Polygon, Coinbase)
    LIVE_DELAYED = "live_delayed"          # Live but potentially delayed (Finnhub)
    CACHED_FRESH = "cached_fresh"          # Cached but recent (FreeCurrency, MT5)
    CACHED_STALE = "cached_stale"          # Cached and potentially stale (AlphaVantage, ExchangeRate)
    MOCK = "mock"                          # Test/mock data

@dataclass
class ProviderConfig:
    """Configuration for a single data provider"""
    name: str
    provider_type: ProviderType
    asset_classes: List[str]              # ['forex', 'crypto', 'metals_oil']
    priority: int                         # Lower = higher priority (1 = highest)
    is_enabled: bool                      # Can be disabled via environment
    requires_api_key: bool                # True if API key required
    api_key_env_var: Optional[str]        # Environment variable name for API key
    timeout_seconds: int                  # Request timeout
    rate_limit_per_minute: Optional[int]  # Rate limit if known
    strict_mode_approved: bool            # Approved for STRICT_LIVE_MODE
    
    def is_available(self) -> bool:
        """Check if provider is available for use"""
        if not self.is_enabled:
            return False
            
        # Check API key if required
        if self.requires_api_key and self.api_key_env_var:
            api_key = os.getenv(self.api_key_env_var)
            if not api_key or api_key.strip() == "":
                return False
                
        return True

class DeterministicProviderConfig:
    """Centralized deterministic provider configuration system"""
    
    def __init__(self):
        self.providers = self._build_provider_configurations()
        self._validate_configuration()
        self._log_configuration()
    
    def _build_provider_configurations(self) -> Dict[str, ProviderConfig]:
        """Build comprehensive provider configuration database"""
        
        # DETERMINISTIC PROVIDER CONFIGURATION
        # Priority order is FIXED and IDENTICAL across all environments
        # This ensures consistent behavior between dev/prod
        
        providers = {
            # === TIER 1: LIVE REAL-TIME PROVIDERS ===
            # Highest priority, best data quality, approved for strict mode
            
            "Coinbase": ProviderConfig(
                name="Coinbase",
                provider_type=ProviderType.LIVE_REAL_TIME,
                asset_classes=["crypto"],
                priority=1,  # Highest priority for crypto
                is_enabled=True,  # No API key required, always enabled
                requires_api_key=False,
                api_key_env_var=None,
                timeout_seconds=10,
                rate_limit_per_minute=100,
                strict_mode_approved=True
            ),
            
            "Polygon.io": ProviderConfig(
                name="Polygon.io",
                provider_type=ProviderType.LIVE_REAL_TIME,
                asset_classes=["forex", "crypto"],
                priority=2,  # High priority, professional grade
                is_enabled=os.getenv('POLYGON_API_KEY', '').strip() != '',
                requires_api_key=True,
                api_key_env_var='POLYGON_API_KEY',
                timeout_seconds=15,
                rate_limit_per_minute=100,
                strict_mode_approved=True
            ),
            
            # === TIER 2: LIVE BUT POTENTIALLY DELAYED ===
            # Good quality, some delay possible
            
            "Binance": ProviderConfig(
                name="Binance",
                provider_type=ProviderType.LIVE_DELAYED,
                asset_classes=["crypto"],
                priority=3,  # Secondary crypto provider
                is_enabled=True,  # No API key required, but may be geo-blocked
                requires_api_key=False,
                api_key_env_var=None,
                timeout_seconds=10,
                rate_limit_per_minute=60,
                strict_mode_approved=True
            ),
            
            "Finnhub": ProviderConfig(
                name="Finnhub",
                provider_type=ProviderType.LIVE_DELAYED,
                asset_classes=["forex", "metals_oil"],  # Added metals_oil support
                priority=4,
                is_enabled=os.getenv('FINNHUB_API_KEY', '').strip() != '',
                requires_api_key=True,
                api_key_env_var='FINNHUB_API_KEY',
                timeout_seconds=10,
                rate_limit_per_minute=60,
                strict_mode_approved=True  # Approved for strict mode including metals_oil
            ),
            
            "CoinGecko": ProviderConfig(
                name="CoinGecko",
                provider_type=ProviderType.LIVE_DELAYED,
                asset_classes=["crypto"],
                priority=5,
                is_enabled=True,  # Free tier available
                requires_api_key=False,
                api_key_env_var=None,
                timeout_seconds=10,
                rate_limit_per_minute=30,
                strict_mode_approved=False  # No OHLC data, only prices
            ),
            
            # === TIER 3: CACHED BUT FRESH ===
            # Reasonable quality, some caching acceptable
            
            "MT5": ProviderConfig(
                name="MT5",
                provider_type=ProviderType.CACHED_FRESH,
                asset_classes=["forex", "metals_oil"],
                priority=6,
                is_enabled=True,  # Local MT5 bridge
                requires_api_key=False,
                api_key_env_var=None,
                timeout_seconds=5,  # Short timeout for local bridge
                rate_limit_per_minute=None,  # No external rate limit
                strict_mode_approved=False  # Bridge timeouts observed
            ),
            
            "FreeCurrencyAPI": ProviderConfig(
                name="FreeCurrencyAPI",
                provider_type=ProviderType.CACHED_FRESH,
                asset_classes=["forex"],
                priority=7,
                is_enabled=os.getenv('FREECURRENCY_API_KEY', '').strip() != '',
                requires_api_key=True,
                api_key_env_var='FREECURRENCY_API_KEY',
                timeout_seconds=10,
                rate_limit_per_minute=100,
                strict_mode_approved=True
            ),
            
            # === TIER 4: CACHED AND POTENTIALLY STALE ===
            # Lower quality, significant caching, use as fallback only
            
            "ExchangeRate.host": ProviderConfig(
                name="ExchangeRate.host",
                provider_type=ProviderType.CACHED_STALE,
                asset_classes=["forex", "crypto", "metals_oil"],
                priority=8,
                is_enabled=True,  # Free, no API key required
                requires_api_key=False,
                api_key_env_var=None,
                timeout_seconds=10,
                rate_limit_per_minute=None,  # No official limit
                strict_mode_approved=False  # Cached data
            ),
            
            "AlphaVantage": ProviderConfig(
                name="AlphaVantage",
                provider_type=ProviderType.CACHED_STALE,
                asset_classes=["forex", "metals_oil"],  # Added metals_oil support
                priority=5,  # Moved up to be after Finnhub for metals_oil fallback
                is_enabled=os.getenv('ALPHAVANTAGE_KEY', '').strip() != '',
                requires_api_key=True,
                api_key_env_var='ALPHAVANTAGE_KEY',
                timeout_seconds=15,
                rate_limit_per_minute=5,  # Very strict rate limit
                strict_mode_approved=True  # Approved for metals_oil commodity data
            ),
            
            # === TIER 5: MOCK/TEST DATA ===
            # Only for development/testing, never in production
            
            "MockDataProvider": ProviderConfig(
                name="MockDataProvider",
                provider_type=ProviderType.MOCK,
                asset_classes=["forex", "crypto", "metals_oil"],
                priority=100,  # Lowest priority
                is_enabled=os.getenv('ENABLE_MOCK_DATA', 'true').lower() == 'true',  # Enabled by default for development
                requires_api_key=False,
                api_key_env_var=None,
                timeout_seconds=1,
                rate_limit_per_minute=None,
                strict_mode_approved=False  # Never approved for strict mode
            )
        }
        
        return providers
    
    def _validate_configuration(self):
        """Validate provider configuration for consistency"""
        # Ensure no priority conflicts within asset classes
        for asset_class in ["forex", "crypto", "metals_oil"]:
            providers_for_class = self.get_providers_for_asset_class(asset_class)
            priorities = [p[1].priority for p in providers_for_class]
            
            if len(priorities) != len(set(priorities)):
                logger.warning(f"Priority conflicts detected for {asset_class} providers")
        
        # Validate API key environment variables
        missing_keys = []
        for provider_config in self.providers.values():
            if provider_config.requires_api_key and provider_config.api_key_env_var:
                if not provider_config.is_available():
                    missing_keys.append((provider_config.name, provider_config.api_key_env_var))
        
        if missing_keys:
            logger.warning(f"Missing API keys for providers: {missing_keys}")
    
    def _log_configuration(self):
        """Log current provider configuration for troubleshooting"""
        logger.info("🔧 DETERMINISTIC PROVIDER CONFIGURATION LOADED")
        
        for asset_class in ["forex", "crypto", "metals_oil"]:
            providers = self.get_providers_for_asset_class(asset_class)
            available_providers = [p for p in providers if p[1].is_available()]
            
            logger.info(f"📊 {asset_class.upper()} PROVIDERS ({len(available_providers)}/{len(providers)} available):")
            
            for provider_instance, config in providers:
                status = "✅ READY" if config.is_available() else "❌ UNAVAILABLE"
                strict_status = "🔒 STRICT-OK" if config.strict_mode_approved else "⚠️ STRICT-BLOCKED"
                
                logger.info(f"   {config.priority:2d}. {config.name:<20} [{config.provider_type.value:<15}] {status} {strict_status}")
                
                if not config.is_available() and config.requires_api_key:
                    logger.info(f"      └─ Missing API key: {config.api_key_env_var}")
    
    def get_providers_for_asset_class(self, asset_class: str) -> List[Tuple[Any, ProviderConfig]]:
        """Get deterministically ordered providers for asset class"""
        # Import providers here to avoid circular imports
        from ..providers.coinbase_provider import CoinbaseProvider
        from ..providers.binance_provider import BinanceProvider
        from ..providers.coingecko_provider import CoinGeckoProvider
        from ..providers.polygon_provider import PolygonProvider
        from ..providers.exchangerate_provider import ExchangeRateProvider
        from ..providers.finnhub_provider import FinnhubProvider
        from ..providers.mt5_data import MT5DataProvider
        from ..providers.freecurrency import FreeCurrencyAPIProvider
        from ..providers.alphavantage import AlphaVantageProvider
        from ..providers.mock import MockDataProvider
        
        # Provider instance mapping
        provider_instances = {
            "Coinbase": CoinbaseProvider(),
            "Binance": BinanceProvider(),
            "CoinGecko": CoinGeckoProvider(),
            "Polygon.io": PolygonProvider(),
            "ExchangeRate.host": ExchangeRateProvider(),
            "Finnhub": FinnhubProvider(),
            "MT5": MT5DataProvider(),
            "FreeCurrencyAPI": FreeCurrencyAPIProvider(),
            "AlphaVantage": AlphaVantageProvider(),
            "MockDataProvider": MockDataProvider()
        }
        
        # Get providers that support this asset class
        compatible_providers = []
        for provider_name, config in self.providers.items():
            if asset_class in config.asset_classes:
                if provider_name in provider_instances:
                    compatible_providers.append((provider_instances[provider_name], config))
        
        # Sort by priority (lower priority number = higher precedence)
        compatible_providers.sort(key=lambda x: x[1].priority)
        
        return compatible_providers
    
    def get_approved_providers_for_asset_class(self, asset_class: str, strict_mode: bool = False) -> List[Tuple[Any, ProviderConfig]]:
        """Get only approved and available providers for asset class"""
        all_providers = self.get_providers_for_asset_class(asset_class)
        
        approved_providers = []
        for provider_instance, config in all_providers:
            # Check availability
            if not config.is_available():
                continue
            
            # Check strict mode approval if required
            if strict_mode and not config.strict_mode_approved:
                continue
            
            # Skip mock providers in strict mode
            if strict_mode and config.provider_type == ProviderType.MOCK:
                continue
                
            approved_providers.append((provider_instance, config))
        
        return approved_providers
    
    def get_provider_config(self, provider_name: str) -> Optional[ProviderConfig]:
        """Get configuration for a specific provider"""
        return self.providers.get(provider_name)
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get complete configuration summary for debugging"""
        summary = {
            'total_providers': len(self.providers),
            'available_providers': len([p for p in self.providers.values() if p.is_available()]),
            'strict_approved_providers': len([p for p in self.providers.values() if p.strict_mode_approved and p.is_available()]),
            'provider_details': {}
        }
        
        for asset_class in ["forex", "crypto", "metals_oil"]:
            providers = self.get_providers_for_asset_class(asset_class)
            available = len([p for p in providers if p[1].is_available()])
            strict_approved = len([p for p in providers if p[1].is_available() and p[1].strict_mode_approved])
            
            summary['provider_details'][asset_class] = {
                'total': len(providers),
                'available': available,
                'strict_approved': strict_approved,
                'provider_order': [p[1].name for p in providers]
            }
        
        return summary

# Global deterministic provider configuration instance
deterministic_provider_config = DeterministicProviderConfig()