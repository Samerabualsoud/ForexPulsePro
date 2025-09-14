#!/usr/bin/env python3
"""
Environment Configuration Validation Script
Ensures identical STRICT_LIVE_MODE settings across development and production environments
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from config.strict_live_config import StrictLiveConfig
    from config.provider_validation import get_provider_validation_service
    from logs.logger import get_logger
except ImportError as e:
    print(f"Error importing backend modules: {e}")
    print("Please run this script from the project root directory")
    sys.exit(1)

logger = get_logger(__name__)

@dataclass
class EnvironmentComparison:
    """Result of environment configuration comparison"""
    environments_match: bool
    differences: List[Dict[str, Any]]
    warnings: List[str]
    critical_issues: List[str]
    environment_fingerprints: Dict[str, str]
    detailed_comparison: Dict[str, Any]

class EnvironmentValidator:
    """Validates and compares environment configurations for parity"""
    
    # Critical settings that MUST be identical across environments
    CRITICAL_STRICT_MODE_SETTINGS = [
        'STRICT_LIVE_MODE',
        'STRICT_LIVE_MAX_DATA_AGE',
        'STRICT_LIVE_MIN_PROVIDERS',
        'STRICT_LIVE_REQUIRE_LIVE_SOURCE',
        'STRICT_LIVE_BLOCK_SYNTHETIC',
        'STRICT_LIVE_BLOCK_MOCK',
        'STRICT_LIVE_BLOCK_CACHED',
        'STRICT_LIVE_REQUIRE_REAL_MARKER',
        'STRICT_LIVE_APPROVED_SOURCES',
        'STRICT_LIVE_BLOCKED_SOURCES',
        'STRICT_LIVE_REQUIRE_MARKET_OPEN',
        'STRICT_LIVE_MIN_DATA_BARS'
    ]
    
    # Settings that can differ between environments
    ALLOWED_DIFFERENCES = [
        'CORS_ORIGINS',
        'LOG_LEVEL',
        'DEBUG_MODE',
        'ENABLE_TEST_ENDPOINTS',
        'DEFAULT_PROVIDER_TIMEOUT',
        'POLYGON_TIMEOUT',
        'FINNHUB_TIMEOUT',
        'MT5_TIMEOUT',
        'ENVIRONMENT',
        'NODE_ENV'
    ]
    
    def __init__(self):
        self.validation_service = get_provider_validation_service()
    
    def load_env_file(self, env_file_path: str) -> Dict[str, str]:
        """Load environment variables from .env file"""
        env_vars = {}
        
        if not os.path.exists(env_file_path):
            raise FileNotFoundError(f"Environment file not found: {env_file_path}")
        
        with open(env_file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                
                # Parse KEY=VALUE format
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    
                    env_vars[key] = value
                else:
                    print(f"Warning: Invalid line format in {env_file_path}:{line_num}: {line}")
        
        return env_vars
    
    def get_current_environment_config(self) -> Dict[str, str]:
        """Get current environment configuration from os.environ"""
        current_config = {}
        
        # Get all relevant environment variables
        all_relevant_vars = (
            self.CRITICAL_STRICT_MODE_SETTINGS + 
            self.ALLOWED_DIFFERENCES +
            ['POLYGON_API_KEY', 'FINNHUB_API_KEY', 'FREECURRENCY_API_KEY', 'ALPHAVANTAGE_KEY']
        )
        
        for var in all_relevant_vars:
            value = os.getenv(var, '')
            current_config[var] = value
        
        return current_config
    
    def validate_strict_mode_configuration(self, env_config: Dict[str, str]) -> Dict[str, Any]:
        """Validate strict mode configuration for a specific environment"""
        validation_result = {
            'environment_name': env_config.get('ENVIRONMENT', 'unknown'),
            'strict_mode_enabled': env_config.get('STRICT_LIVE_MODE', '').lower() == 'true',
            'configuration_valid': True,
            'issues': [],
            'warnings': [],
            'strict_mode_settings': {}
        }
        
        # Extract all strict mode settings
        for setting in self.CRITICAL_STRICT_MODE_SETTINGS:
            value = env_config.get(setting, '')
            validation_result['strict_mode_settings'][setting] = value
            
            # Validate required settings
            if not value and setting in ['STRICT_LIVE_MODE']:
                validation_result['issues'].append(f"Missing required setting: {setting}")
                validation_result['configuration_valid'] = False
        
        # Validate logical consistency
        if validation_result['strict_mode_enabled']:
            # Check for conflicting settings
            if env_config.get('ENABLE_MOCK_DATA', '').lower() == 'true':
                validation_result['issues'].append("Conflicting configuration: STRICT_LIVE_MODE enabled with ENABLE_MOCK_DATA")
                validation_result['configuration_valid'] = False
            
            # Check that approved sources are defined
            approved_sources = env_config.get('STRICT_LIVE_APPROVED_SOURCES', '')
            if not approved_sources:
                validation_result['warnings'].append("No approved sources defined for strict mode")
        
        return validation_result
    
    def compare_environments(self, env1_config: Dict[str, str], env2_config: Dict[str, str], 
                           env1_name: str = "Environment 1", env2_name: str = "Environment 2") -> EnvironmentComparison:
        """Compare two environment configurations for parity"""
        
        differences = []
        warnings = []
        critical_issues = []
        
        # Check critical strict mode settings
        for setting in self.CRITICAL_STRICT_MODE_SETTINGS:
            value1 = env1_config.get(setting, '')
            value2 = env2_config.get(setting, '')
            
            if value1 != value2:
                difference = {
                    'setting': setting,
                    'environment_1': {env1_name: value1},
                    'environment_2': {env2_name: value2},
                    'severity': 'critical'
                }
                differences.append(difference)
                critical_issues.append(f"Critical difference in {setting}: {env1_name}='{value1}' vs {env2_name}='{value2}'")
        
        # Check for allowed differences (just for informational purposes)
        for setting in self.ALLOWED_DIFFERENCES:
            value1 = env1_config.get(setting, '')
            value2 = env2_config.get(setting, '')
            
            if value1 != value2:
                difference = {
                    'setting': setting,
                    'environment_1': {env1_name: value1},
                    'environment_2': {env2_name: value2},
                    'severity': 'allowed'
                }
                differences.append(difference)
        
        # Generate environment fingerprints
        fingerprints = {}
        for env_name, config in [(env1_name, env1_config), (env2_name, env2_config)]:
            # Create a temporary environment with these settings
            original_env = {}
            for key, value in config.items():
                original_env[key] = os.getenv(key)
                os.environ[key] = value
            
            try:
                # Generate fingerprint
                fingerprint = self.validation_service.generate_configuration_fingerprint()
                fingerprints[env_name] = fingerprint
            finally:
                # Restore original environment
                for key, original_value in original_env.items():
                    if original_value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = original_value
        
        # Overall assessment
        environments_match = len(critical_issues) == 0
        
        detailed_comparison = {
            'critical_strict_mode_settings': {
                setting: {
                    env1_name: env1_config.get(setting, ''),
                    env2_name: env2_config.get(setting, ''),
                    'match': env1_config.get(setting, '') == env2_config.get(setting, '')
                }
                for setting in self.CRITICAL_STRICT_MODE_SETTINGS
            },
            'allowed_differences': {
                setting: {
                    env1_name: env1_config.get(setting, ''),
                    env2_name: env2_config.get(setting, ''),
                    'differs': env1_config.get(setting, '') != env2_config.get(setting, '')
                }
                for setting in self.ALLOWED_DIFFERENCES
            }
        }
        
        return EnvironmentComparison(
            environments_match=environments_match,
            differences=differences,
            warnings=warnings,
            critical_issues=critical_issues,
            environment_fingerprints=fingerprints,
            detailed_comparison=detailed_comparison
        )
    
    def generate_parity_report(self, comparison: EnvironmentComparison) -> str:
        """Generate a comprehensive parity report"""
        report_lines = [
            "=" * 80,
            "ENVIRONMENT PARITY VALIDATION REPORT",
            "=" * 80,
            f"Generated at: {json.dumps(str(__import__('datetime').datetime.now()))}",
            "",
            "OVERALL ASSESSMENT:",
            f"✅ Environments Match: {comparison.environments_match}" if comparison.environments_match else f"❌ Environments Match: {comparison.environments_match}",
            f"Critical Issues: {len(comparison.critical_issues)}",
            f"Total Differences: {len(comparison.differences)}",
            f"Warnings: {len(comparison.warnings)}",
            "",
            "ENVIRONMENT FINGERPRINTS:",
        ]
        
        for env_name, fingerprint in comparison.environment_fingerprints.items():
            report_lines.append(f"  {env_name}: {fingerprint}")
        
        if comparison.critical_issues:
            report_lines.extend([
                "",
                "🚨 CRITICAL ISSUES (MUST BE FIXED):",
                "=" * 40
            ])
            for issue in comparison.critical_issues:
                report_lines.append(f"  ❌ {issue}")
        
        if comparison.differences:
            report_lines.extend([
                "",
                "DETAILED DIFFERENCES:",
                "=" * 40
            ])
            for diff in comparison.differences:
                severity_icon = "🚨" if diff['severity'] == 'critical' else "ℹ️"
                report_lines.append(f"  {severity_icon} {diff['setting']} ({diff['severity']}):")
                for env_key, env_val in diff['environment_1'].items():
                    report_lines.append(f"    {env_key}: '{env_val}'")
                for env_key, env_val in diff['environment_2'].items():
                    report_lines.append(f"    {env_key}: '{env_val}'")
                report_lines.append("")
        
        if comparison.warnings:
            report_lines.extend([
                "",
                "⚠️ WARNINGS:",
                "=" * 40
            ])
            for warning in comparison.warnings:
                report_lines.append(f"  ⚠️ {warning}")
        
        if comparison.environments_match:
            report_lines.extend([
                "",
                "✅ ENVIRONMENT PARITY CONFIRMED",
                "Both environments have identical STRICT_LIVE_MODE configurations.",
                "Signal generation behavior will be consistent across environments.",
                ""
            ])
        else:
            report_lines.extend([
                "",
                "❌ ENVIRONMENT PARITY ISSUES DETECTED",
                "Differences in critical settings may cause inconsistent behavior.",
                "Please resolve all critical issues before deployment.",
                ""
            ])
        
        report_lines.extend([
            "=" * 80,
            "END OF REPORT",
            "=" * 80
        ])
        
        return "\n".join(report_lines)

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Validate environment configuration parity")
    parser.add_argument('--env1', type=str, help="Path to first environment file")
    parser.add_argument('--env2', type=str, help="Path to second environment file")
    parser.add_argument('--current', action='store_true', help="Compare current environment with a file")
    parser.add_argument('--output', type=str, help="Output file for the report")
    parser.add_argument('--quiet', action='store_true', help="Only show critical issues")
    
    args = parser.parse_args()
    
    validator = EnvironmentValidator()
    
    try:
        if args.current and args.env1:
            # Compare current environment with file
            current_config = validator.get_current_environment_config()
            file_config = validator.load_env_file(args.env1)
            comparison = validator.compare_environments(
                current_config, file_config, 
                "Current Environment", f"File: {args.env1}"
            )
        elif args.env1 and args.env2:
            # Compare two files
            config1 = validator.load_env_file(args.env1)
            config2 = validator.load_env_file(args.env2)
            comparison = validator.compare_environments(
                config1, config2,
                f"File: {args.env1}", f"File: {args.env2}"
            )
        else:
            print("Error: Please specify either --current with --env1, or both --env1 and --env2")
            sys.exit(1)
        
        # Generate report
        report = validator.generate_parity_report(comparison)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"Report saved to: {args.output}")
        
        if not args.quiet:
            print(report)
        else:
            # Only show critical issues in quiet mode
            if comparison.critical_issues:
                print("CRITICAL ISSUES DETECTED:")
                for issue in comparison.critical_issues:
                    print(f"  ❌ {issue}")
            else:
                print("✅ No critical issues detected")
        
        # Exit with error code if environments don't match
        sys.exit(0 if comparison.environments_match else 1)
        
    except Exception as e:
        print(f"Error during validation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()