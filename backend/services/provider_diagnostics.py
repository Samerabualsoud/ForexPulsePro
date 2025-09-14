"""
Provider Diagnostics Service

Comprehensive diagnostics for provider status and configuration verification.
Helps ensure identical behavior between development and production environments.
"""

import os
import hashlib
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import asdict

from ..config.provider_config import deterministic_provider_config, ProviderConfig, ProviderType
from ..config.strict_live_config import StrictLiveConfig
from ..logs.logger import get_logger

logger = get_logger(__name__)


class ProviderDiagnosticsService:
    """Service for comprehensive provider diagnostics and health checking"""
    
    def __init__(self):
        self.provider_config = deterministic_provider_config
        self.timestamp = datetime.utcnow()
    
    def get_comprehensive_diagnostics(self) -> Dict[str, Any]:
        """Get complete provider diagnostics for environment comparison"""
        logger.info("🔍 Generating comprehensive provider diagnostics...")
        
        diagnostics = {
            "timestamp": self.timestamp.isoformat(),
            "environment": self._detect_environment(),
            "configuration_fingerprint": self._generate_configuration_fingerprint(),
            "strict_live_mode": self._get_strict_mode_diagnostics(),
            "provider_overview": self._get_provider_overview(),
            "provider_details": self._get_detailed_provider_status(),
            "asset_class_routing": self._get_asset_class_routing(),
            "health_checks": self._perform_provider_health_checks(),
            "environment_variables": self._get_relevant_environment_variables(),
            "configuration_warnings": self._validate_configuration(),
            "troubleshooting": self._generate_troubleshooting_info()
        }
        
        logger.info(f"✅ Diagnostics generated with fingerprint: {diagnostics['configuration_fingerprint']}")
        return diagnostics
    
    def _detect_environment(self) -> Dict[str, Any]:
        """Detect current environment and deployment characteristics"""
        return {
            "type": os.getenv('ENVIRONMENT', 'development').lower(),
            "strict_live_mode_enabled": StrictLiveConfig.ENABLED,
            "backend_host": os.getenv('BACKEND_HOST', 'localhost'),
            "backend_port": os.getenv('BACKEND_PORT', '8080'),
            "cors_origins": os.getenv('CORS_ORIGINS', '*'),
            "deployment_timestamp": os.getenv('DEPLOYMENT_TIMESTAMP', 'unknown'),
            "python_version": os.getenv('PYTHON_VERSION', 'unknown')
        }
    
    def _generate_configuration_fingerprint(self) -> str:
        """Generate unique fingerprint for provider configuration comparison"""
        # Create deterministic configuration snapshot
        config_data = {
            "provider_configs": {},
            "strict_mode_settings": StrictLiveConfig.get_status_summary(),
            "environment_type": os.getenv('ENVIRONMENT', 'development').lower()
        }
        
        # Add provider configurations (excluding sensitive data)
        for name, config in self.provider_config.providers.items():
            config_data["provider_configs"][name] = {
                "provider_type": config.provider_type.value,
                "asset_classes": config.asset_classes,
                "priority": config.priority,
                "is_enabled": config.is_enabled,
                "requires_api_key": config.requires_api_key,
                "timeout_seconds": config.timeout_seconds,
                "rate_limit_per_minute": config.rate_limit_per_minute,
                "strict_mode_approved": config.strict_mode_approved
            }
        
        # Generate SHA256 hash of configuration
        config_json = json.dumps(config_data, sort_keys=True)
        fingerprint = hashlib.sha256(config_json.encode()).hexdigest()[:16]
        
        return fingerprint
    
    def _get_strict_mode_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive strict live mode configuration"""
        status_summary = StrictLiveConfig.get_status_summary()
        
        # Add validation results
        validation_results = {}
        for provider_name in self.provider_config.providers.keys():
            is_approved = StrictLiveConfig.is_data_source_approved(provider_name)
            is_blocked = StrictLiveConfig.is_data_source_blocked(provider_name)
            
            validation_results[provider_name] = {
                "approved": is_approved,
                "blocked": is_blocked,
                "status": "approved" if is_approved and not is_blocked else "blocked"
            }
        
        return {
            **status_summary,
            "provider_validation": validation_results
        }
    
    def _get_provider_overview(self) -> Dict[str, Any]:
        """Get high-level provider statistics"""
        total_providers = len(self.provider_config.providers)
        available_providers = sum(1 for p in self.provider_config.providers.values() if p.is_available())
        strict_approved = sum(1 for p in self.provider_config.providers.values() 
                             if p.strict_mode_approved and p.is_available())
        
        # Count by provider type
        type_counts = {}
        for provider_type in ProviderType:
            count = sum(1 for p in self.provider_config.providers.values() 
                       if p.provider_type == provider_type and p.is_available())
            type_counts[provider_type.value] = count
        
        # Count by asset class
        asset_class_counts = {}
        for asset_class in ["forex", "crypto", "metals_oil"]:
            available_for_class = len(self.provider_config.get_approved_providers_for_asset_class(asset_class))
            strict_approved_for_class = len(self.provider_config.get_approved_providers_for_asset_class(
                asset_class, strict_mode=True))
            
            asset_class_counts[asset_class] = {
                "available": available_for_class,
                "strict_approved": strict_approved_for_class
            }
        
        return {
            "total_providers": total_providers,
            "available_providers": available_providers,
            "unavailable_providers": total_providers - available_providers,
            "strict_approved_providers": strict_approved,
            "provider_type_distribution": type_counts,
            "asset_class_availability": asset_class_counts
        }
    
    def _get_detailed_provider_status(self) -> Dict[str, Any]:
        """Get detailed status for each provider"""
        provider_details = {}
        
        for name, config in self.provider_config.providers.items():
            # Check API key status
            api_key_status = "not_required"
            if config.requires_api_key:
                if config.api_key_env_var:
                    api_key = os.getenv(config.api_key_env_var, '')
                    if api_key.strip():
                        api_key_status = "present"
                    else:
                        api_key_status = "missing"
                else:
                    api_key_status = "no_env_var_configured"
            
            provider_details[name] = {
                "configuration": {
                    "provider_type": config.provider_type.value,
                    "asset_classes": config.asset_classes,
                    "priority": config.priority,
                    "timeout_seconds": config.timeout_seconds,
                    "rate_limit_per_minute": config.rate_limit_per_minute,
                    "strict_mode_approved": config.strict_mode_approved
                },
                "status": {
                    "is_enabled": config.is_enabled,
                    "is_available": config.is_available(),
                    "api_key_status": api_key_status,
                    "api_key_env_var": config.api_key_env_var,
                    "strict_mode_compatible": StrictLiveConfig.is_data_source_approved(name)
                },
                "health": {
                    "last_checked": None,  # Will be populated by health checks
                    "connectivity": "unknown",
                    "response_time_ms": None,
                    "error_details": None
                }
            }
        
        return provider_details
    
    def _get_asset_class_routing(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get provider routing order for each asset class"""
        routing = {}
        
        for asset_class in ["forex", "crypto", "metals_oil"]:
            # Get all providers for asset class
            all_providers = self.provider_config.get_providers_for_asset_class(asset_class)
            strict_providers = self.provider_config.get_approved_providers_for_asset_class(
                asset_class, strict_mode=True)
            
            routing_info = []
            for provider_instance, config in all_providers:
                is_strict_approved = any(p[1].name == config.name for p in strict_providers)
                
                routing_info.append({
                    "provider_name": config.name,
                    "priority": config.priority,
                    "provider_type": config.provider_type.value,
                    "is_available": config.is_available(),
                    "strict_mode_approved": is_strict_approved,
                    "will_be_used_normal": config.is_available(),
                    "will_be_used_strict": is_strict_approved and config.is_available()
                })
            
            routing[asset_class] = routing_info
        
        return routing
    
    def _perform_provider_health_checks(self) -> Dict[str, Any]:
        """Perform basic health checks on available providers"""
        health_results = {
            "summary": {
                "total_checked": 0,
                "healthy": 0,
                "unhealthy": 0,
                "timeout": 0
            },
            "details": {}
        }
        
        for name, config in self.provider_config.providers.items():
            if not config.is_available():
                continue
            
            health_results["summary"]["total_checked"] += 1
            check_result = self._check_provider_health(name, config)
            health_results["details"][name] = check_result
            
            if check_result["status"] == "healthy":
                health_results["summary"]["healthy"] += 1
            elif check_result["status"] == "timeout":
                health_results["summary"]["timeout"] += 1
            else:
                health_results["summary"]["unhealthy"] += 1
        
        return health_results
    
    def _check_provider_health(self, provider_name: str, config: ProviderConfig) -> Dict[str, Any]:
        """Perform basic health check on a single provider"""
        try:
            start_time = time.time()
            
            # Basic connectivity check - this is a simplified version
            # In a full implementation, you would actually test the provider's API
            
            # Simulate health check based on configuration
            if config.provider_type == ProviderType.MOCK:
                return {
                    "status": "healthy",
                    "response_time_ms": 1,
                    "message": "Mock provider always healthy",
                    "checked_at": datetime.utcnow().isoformat()
                }
            
            # For real providers, we'd need to make actual API calls
            # For now, return a simulated health status
            response_time = time.time() - start_time
            
            return {
                "status": "unknown",  # Would be "healthy", "unhealthy", or "timeout"
                "response_time_ms": int(response_time * 1000),
                "message": "Health check not implemented for this provider type",
                "checked_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "response_time_ms": None,
                "message": f"Health check failed: {str(e)}",
                "checked_at": datetime.utcnow().isoformat()
            }
    
    def _get_relevant_environment_variables(self) -> Dict[str, Any]:
        """Get relevant environment variables for provider configuration"""
        env_vars = {
            "api_keys": {},
            "configuration": {},
            "system": {}
        }
        
        # API key environment variables
        for config in self.provider_config.providers.values():
            if config.api_key_env_var:
                value = os.getenv(config.api_key_env_var, '')
                env_vars["api_keys"][config.api_key_env_var] = {
                    "is_set": bool(value.strip()),
                    "value_length": len(value.strip()) if value.strip() else 0,
                    "provider": config.name
                }
        
        # Configuration environment variables
        config_vars = [
            'STRICT_LIVE_MODE', 'STRICT_LIVE_MAX_DATA_AGE', 'STRICT_LIVE_MIN_PROVIDERS',
            'STRICT_LIVE_REQUIRE_LIVE_SOURCE', 'STRICT_LIVE_APPROVED_SOURCES',
            'STRICT_LIVE_BLOCKED_SOURCES', 'ENABLE_MOCK_DATA'
        ]
        
        for var in config_vars:
            value = os.getenv(var)
            env_vars["configuration"][var] = {
                "value": value,
                "is_set": value is not None
            }
        
        # System environment variables
        system_vars = ['ENVIRONMENT', 'BACKEND_HOST', 'BACKEND_PORT', 'CORS_ORIGINS']
        for var in system_vars:
            env_vars["system"][var] = os.getenv(var, 'not_set')
        
        return env_vars
    
    def _validate_configuration(self) -> List[Dict[str, Any]]:
        """Validate configuration and return warnings/issues"""
        warnings = []
        
        # Check for missing API keys for enabled providers
        for name, config in self.provider_config.providers.items():
            if config.is_enabled and config.requires_api_key and not config.is_available():
                warnings.append({
                    "type": "missing_api_key",
                    "severity": "warning",
                    "provider": name,
                    "message": f"Provider {name} is enabled but missing API key: {config.api_key_env_var}",
                    "suggestion": f"Set environment variable {config.api_key_env_var} to enable this provider"
                })
        
        # Check strict mode configuration
        if StrictLiveConfig.ENABLED:
            strict_approved_count = sum(1 for p in self.provider_config.providers.values() 
                                      if p.strict_mode_approved and p.is_available())
            
            if strict_approved_count == 0:
                warnings.append({
                    "type": "no_strict_providers",
                    "severity": "error",
                    "message": "Strict live mode is enabled but no approved providers are available",
                    "suggestion": "Configure API keys for approved providers or disable strict mode"
                })
            elif strict_approved_count < 2:
                warnings.append({
                    "type": "insufficient_strict_providers",
                    "severity": "warning",
                    "message": f"Only {strict_approved_count} strict-approved provider available",
                    "suggestion": "Configure additional approved providers for redundancy"
                })
        
        # Check for asset classes with no providers
        for asset_class in ["forex", "crypto", "metals_oil"]:
            available_count = len(self.provider_config.get_approved_providers_for_asset_class(asset_class))
            if available_count == 0:
                warnings.append({
                    "type": "no_providers_for_asset_class",
                    "severity": "warning",
                    "asset_class": asset_class,
                    "message": f"No available providers for {asset_class} assets",
                    "suggestion": f"Configure API keys for {asset_class} providers"
                })
        
        return warnings
    
    def _generate_troubleshooting_info(self) -> Dict[str, Any]:
        """Generate troubleshooting information and common fixes"""
        return {
            "common_issues": [
                {
                    "issue": "Providers hitting rate limits",
                    "symptoms": ["429 HTTP errors", "Provider timeouts", "Signal generation failures"],
                    "solutions": [
                        "Configure additional API keys for redundancy",
                        "Upgrade to higher rate limit plans",
                        "Implement request caching",
                        "Add request delays"
                    ]
                },
                {
                    "issue": "Strict mode blocking all signals",
                    "symptoms": ["No signals generated", "All providers blocked", "Safety mode errors"],
                    "solutions": [
                        "Configure approved provider API keys",
                        "Verify provider connectivity",
                        "Check STRICT_LIVE_APPROVED_SOURCES setting",
                        "Temporarily disable strict mode for testing"
                    ]
                },
                {
                    "issue": "Environment configuration mismatch",
                    "symptoms": ["Different behavior between dev/prod", "Missing providers in production"],
                    "solutions": [
                        "Compare configuration fingerprints",
                        "Verify environment variables match",
                        "Check API key availability in both environments",
                        "Use this diagnostics endpoint to compare"
                    ]
                }
            ],
            "configuration_fingerprint_usage": {
                "purpose": "Compare environments for identical configuration",
                "how_to_use": [
                    "Call this endpoint in development environment",
                    "Call this endpoint in production environment", 
                    "Compare configuration_fingerprint values",
                    "If fingerprints match, configurations are identical",
                    "If different, investigate provider and environment variable differences"
                ]
            }
        }


# Global instance for use across the application
provider_diagnostics_service = ProviderDiagnosticsService()