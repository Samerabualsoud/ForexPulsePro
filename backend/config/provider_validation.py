"""
Provider Validation and Configuration Consistency System
Ensures identical provider behavior between development and production environments
"""

import os
import hashlib
import json
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import asdict
from ..logs.logger import get_logger

logger = get_logger(__name__)

class ProviderValidationService:
    """Service for validating provider configurations and ensuring consistency"""
    
    def __init__(self, provider_config):
        self.provider_config = provider_config
    
    def generate_configuration_fingerprint(self) -> str:
        """Generate a unique fingerprint for the current provider configuration"""
        config_summary = self.provider_config.get_configuration_summary()
        
        # Create deterministic configuration data for hashing
        fingerprint_data = {
            'total_providers': config_summary['total_providers'],
            'available_providers': config_summary['available_providers'],
            'strict_approved_providers': config_summary['strict_approved_providers'],
            'provider_details': {}
        }
        
        # Add deterministic provider order and configuration for each asset class
        for asset_class in ['forex', 'crypto', 'metals_oil']:
            providers = self.provider_config.get_approved_providers_for_asset_class(asset_class, strict_mode=False)
            provider_fingerprints = []
            
            for provider_instance, config in providers:
                provider_fingerprint = {
                    'name': config.name,
                    'provider_type': config.provider_type.value,
                    'priority': config.priority,
                    'is_enabled': config.is_enabled,
                    'requires_api_key': config.requires_api_key,
                    'api_key_available': bool(os.getenv(config.api_key_env_var, '').strip()) if config.api_key_env_var else True,
                    'timeout_seconds': config.timeout_seconds,
                    'strict_mode_approved': config.strict_mode_approved,
                    'is_available': config.is_available()
                }
                provider_fingerprints.append(provider_fingerprint)
            
            fingerprint_data['provider_details'][asset_class] = {
                'total_providers': len(providers),
                'available_providers': len([p for p in provider_fingerprints if p['is_available']]),
                'strict_approved': len([p for p in provider_fingerprints if p['strict_mode_approved'] and p['is_available']]),
                'provider_order': provider_fingerprints
            }
        
        # Generate deterministic hash
        config_json = json.dumps(fingerprint_data, sort_keys=True)
        fingerprint = hashlib.sha256(config_json.encode()).hexdigest()[:16]
        
        return fingerprint
    
    def validate_environment_configuration(self) -> Dict[str, Any]:
        """Validate the current environment's provider configuration"""
        validation_result = {
            'environment': os.getenv('ENVIRONMENT', 'unknown'),
            'strict_mode_enabled': os.getenv('STRICT_LIVE_MODE', 'true').lower() == 'true',
            'configuration_fingerprint': self.generate_configuration_fingerprint(),
            'timestamp': str(pd.Timestamp.now()),
            'validation_passed': True,
            'issues': [],
            'warnings': [],
            'provider_status': {},
            'environment_variables': self._get_relevant_environment_variables()
        }
        
        # Validate each asset class
        for asset_class in ['forex', 'crypto', 'metals_oil']:
            asset_validation = self._validate_asset_class_providers(asset_class)
            validation_result['provider_status'][asset_class] = asset_validation
            
            # Collect issues and warnings
            if asset_validation['issues']:
                validation_result['issues'].extend(asset_validation['issues'])
                validation_result['validation_passed'] = False
            
            if asset_validation['warnings']:
                validation_result['warnings'].extend(asset_validation['warnings'])
        
        # Check for critical configuration issues
        critical_issues = self._check_critical_configuration_issues()
        if critical_issues:
            validation_result['issues'].extend(critical_issues)
            validation_result['validation_passed'] = False
        
        return validation_result
    
    def _validate_asset_class_providers(self, asset_class: str) -> Dict[str, Any]:
        """Validate providers for a specific asset class"""
        providers = self.provider_config.get_providers_for_asset_class(asset_class)
        approved_providers = self.provider_config.get_approved_providers_for_asset_class(asset_class, strict_mode=False)
        strict_approved = self.provider_config.get_approved_providers_for_asset_class(asset_class, strict_mode=True)
        
        validation = {
            'total_providers': len(providers),
            'available_providers': len(approved_providers),
            'strict_approved_providers': len(strict_approved),
            'issues': [],
            'warnings': [],
            'provider_details': []
        }
        
        # Check if we have any available providers
        if not approved_providers:
            validation['issues'].append(f"No available providers for {asset_class}")
        
        # Check if we have strict-mode approved providers when strict mode is enabled
        if os.getenv('STRICT_LIVE_MODE', 'true').lower() == 'true' and not strict_approved:
            validation['issues'].append(f"No strict-mode approved providers for {asset_class}")
        
        # Validate each provider
        for provider_instance, config in providers:
            provider_validation = {
                'name': config.name,
                'priority': config.priority,
                'is_enabled': config.is_enabled,
                'is_available': config.is_available(),
                'strict_approved': config.strict_mode_approved,
                'issues': [],
                'warnings': []
            }
            
            # Check API key requirements
            if config.requires_api_key and config.api_key_env_var:
                api_key = os.getenv(config.api_key_env_var, '').strip()
                if not api_key:
                    provider_validation['issues'].append(f"Missing required API key: {config.api_key_env_var}")
                elif api_key == "your_api_key_here" or len(api_key) < 10:
                    provider_validation['warnings'].append(f"Potentially invalid API key for {config.api_key_env_var}")
            
            # Check for known provider issues
            if config.name == 'Binance' and config.is_enabled:
                provider_validation['warnings'].append("Binance may be geo-blocked in some regions")
            
            if config.name == 'AlphaVantage' and config.is_enabled:
                provider_validation['warnings'].append("AlphaVantage has strict rate limits (5 calls/minute)")
            
            validation['provider_details'].append(provider_validation)
            
            # Collect issues and warnings
            if provider_validation['issues']:
                validation['issues'].extend(provider_validation['issues'])
            if provider_validation['warnings']:
                validation['warnings'].extend(provider_validation['warnings'])
        
        return validation
    
    def _check_critical_configuration_issues(self) -> List[str]:
        """Check for critical configuration issues that could cause environment differences"""
        issues = []
        
        # Check environment variable consistency
        critical_env_vars = [
            'STRICT_LIVE_MODE',
            'POLYGON_API_KEY', 
            'FINNHUB_API_KEY',
            'ALPHAVANTAGE_API_KEY',
            'FREECURRENCY_API_KEY',
            'ENABLE_MOCK_DATA'
        ]
        
        for env_var in critical_env_vars:
            value = os.getenv(env_var, '').strip()
            if not value and env_var != 'ENABLE_MOCK_DATA':
                # Only warn for missing optional API keys, not error
                pass  # Will be caught in provider validation
        
        # Check for conflicting configurations
        if os.getenv('STRICT_LIVE_MODE', 'true').lower() == 'true' and os.getenv('ENABLE_MOCK_DATA', 'false').lower() == 'true':
            issues.append("Conflicting configuration: STRICT_LIVE_MODE enabled with ENABLE_MOCK_DATA")
        
        return issues
    
    def _get_relevant_environment_variables(self) -> Dict[str, str]:
        """Get relevant environment variables for configuration comparison"""
        relevant_vars = [
            'ENVIRONMENT',
            'STRICT_LIVE_MODE',
            'POLYGON_API_KEY',
            'FINNHUB_API_KEY', 
            'ALPHAVANTAGE_API_KEY',
            'FREECURRENCY_API_KEY',
            'ENABLE_MOCK_DATA',
            'ENABLE_DUAL_AI',
            'AI_CONSENSUS_THRESHOLD',
            'AUTO_TRADE_ENABLED'
        ]
        
        env_vars = {}
        for var in relevant_vars:
            value = os.getenv(var, '')
            # Mask sensitive values but show if they exist
            if 'API_KEY' in var:
                env_vars[var] = '***MASKED***' if value.strip() else 'NOT_SET'
            else:
                env_vars[var] = value
        
        return env_vars
    
    def compare_with_reference_configuration(self, reference_fingerprint: str) -> Dict[str, Any]:
        """Compare current configuration with a reference configuration"""
        current_fingerprint = self.generate_configuration_fingerprint()
        
        comparison = {
            'matches_reference': current_fingerprint == reference_fingerprint,
            'current_fingerprint': current_fingerprint,
            'reference_fingerprint': reference_fingerprint,
            'timestamp': str(pd.Timestamp.now())
        }
        
        if not comparison['matches_reference']:
            logger.warning(f"Configuration mismatch detected!")
            logger.warning(f"Current: {current_fingerprint}")
            logger.warning(f"Reference: {reference_fingerprint}")
        else:
            logger.info(f"Configuration matches reference: {current_fingerprint}")
        
        return comparison
    
    def generate_configuration_report(self) -> Dict[str, Any]:
        """Generate a comprehensive configuration report for troubleshooting"""
        validation_result = self.validate_environment_configuration()
        
        report = {
            'report_type': 'provider_configuration_report',
            'generated_at': str(pd.Timestamp.now()),
            'environment': validation_result['environment'],
            'configuration_fingerprint': validation_result['configuration_fingerprint'],
            'validation_summary': {
                'validation_passed': validation_result['validation_passed'],
                'total_issues': len(validation_result['issues']),
                'total_warnings': len(validation_result['warnings'])
            },
            'environment_variables': validation_result['environment_variables'],
            'provider_status': validation_result['provider_status'],
            'issues': validation_result['issues'],
            'warnings': validation_result['warnings'],
            'recommendations': self._generate_recommendations(validation_result)
        }
        
        return report
    
    def _generate_recommendations(self, validation_result: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Recommendations for missing API keys
        for issue in validation_result['issues']:
            if 'Missing required API key' in issue:
                recommendations.append(f"Set the missing API key mentioned in: {issue}")
        
        # Recommendations for strict mode issues
        if any('No strict-mode approved providers' in issue for issue in validation_result['issues']):
            recommendations.append("Add API keys for strict-mode approved providers (Polygon.io, FreeCurrencyAPI, Coinbase)")
        
        # Recommendations for rate limiting
        if any('AlphaVantage has strict rate limits' in warning for warning in validation_result['warnings']):
            recommendations.append("Consider using higher-tier API providers to avoid rate limiting issues")
        
        # Environment-specific recommendations
        if not validation_result['validation_passed']:
            recommendations.append("Review provider configuration and ensure API keys are properly set")
            recommendations.append("Check that the same environment variables are set in both dev and production")
        
        return recommendations

# Import pandas here to avoid circular imports
try:
    import pandas as pd
except ImportError:
    # Fallback timestamp if pandas not available
    class pd:
        @staticmethod
        def Timestamp():
            from datetime import datetime
            return datetime.utcnow()
        
        @staticmethod
        def now():
            from datetime import datetime
            return datetime.utcnow()

# Global validation service instance
provider_validation_service = None

def get_provider_validation_service():
    """Get the global provider validation service instance"""
    global provider_validation_service
    if provider_validation_service is None:
        from .provider_config import deterministic_provider_config
        provider_validation_service = ProviderValidationService(deterministic_provider_config)
    return provider_validation_service