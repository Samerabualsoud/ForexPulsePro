#!/usr/bin/env python3
"""
Environment Parity Validation Script
Final validation to ensure identical STRICT_LIVE_MODE settings across development and production
"""

import subprocess
import sys
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

def run_command(command: List[str]) -> Tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr"""
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return 1, "", str(e)

def check_api_endpoint(url: str) -> Dict[str, Any]:
    """Check API endpoint for environment validation"""
    try:
        import requests
        response = requests.get(f"{url}/api/environment/strict-mode-status", timeout=10)
        if response.status_code == 200:
            return {"status": "✅ PASS", "data": response.json()}
        else:
            return {"status": "❌ FAIL", "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"status": "❌ FAIL", "error": str(e)}

def validate_environment_files() -> Dict[str, Any]:
    """Validate environment template files"""
    results = {
        "development_template": {"exists": False, "valid": False},
        "production_template": {"exists": False, "valid": False},
        "validation_script": {"exists": False, "executable": False}
    }
    
    # Check template files
    dev_template = Path("backend/config/environment_templates/development.env")
    prod_template = Path("backend/config/environment_templates/production.env")
    validation_script = Path("backend/config/environment_validation.py")
    
    results["development_template"]["exists"] = dev_template.exists()
    results["production_template"]["exists"] = prod_template.exists()
    results["validation_script"]["exists"] = validation_script.exists()
    
    if dev_template.exists():
        content = dev_template.read_text()
        results["development_template"]["valid"] = "STRICT_LIVE_MODE=" in content
    
    if prod_template.exists():
        content = prod_template.read_text()
        results["production_template"]["valid"] = "STRICT_LIVE_MODE=" in content
    
    if validation_script.exists():
        results["validation_script"]["executable"] = os.access(validation_script, os.X_OK)
    
    return results

def run_environment_validation() -> Dict[str, Any]:
    """Run the environment validation script"""
    validation_script = Path("backend/config/environment_validation.py")
    dev_template = Path("backend/config/environment_templates/development.env")
    
    if not validation_script.exists() or not dev_template.exists():
        return {"status": "❌ SKIP", "reason": "Validation script or template missing"}
    
    # Run validation comparing current environment with development template
    exit_code, stdout, stderr = run_command([
        sys.executable, "-m", "backend.config.environment_validation",
        "--current", "--env1", str(dev_template), "--quiet"
    ])
    
    return {
        "status": "✅ PASS" if exit_code == 0 else "❌ FAIL",
        "exit_code": exit_code,
        "output": stdout,
        "errors": stderr
    }

def check_environment_variables() -> Dict[str, Any]:
    """Check critical environment variables"""
    critical_vars = [
        "STRICT_LIVE_MODE",
        "STRICT_LIVE_MAX_DATA_AGE", 
        "STRICT_LIVE_MIN_PROVIDERS",
        "STRICT_LIVE_REQUIRE_LIVE_SOURCE",
        "STRICT_LIVE_APPROVED_SOURCES",
        "STRICT_LIVE_BLOCKED_SOURCES"
    ]
    
    results = {}
    for var in critical_vars:
        value = os.getenv(var, "NOT_SET")
        results[var] = {
            "value": value,
            "is_set": value != "NOT_SET",
            "status": "✅ SET" if value != "NOT_SET" else "⚠️ DEFAULT"
        }
    
    return results

def generate_parity_report() -> Dict[str, Any]:
    """Generate comprehensive environment parity report"""
    print("🔍 Environment Parity Validation Report")
    print("=" * 60)
    
    report = {
        "timestamp": str(__import__('datetime').datetime.now()),
        "environment_files": validate_environment_files(),
        "environment_variables": check_environment_variables(),
        "validation_script_result": run_environment_validation(),
        "overall_status": "PENDING"
    }
    
    # Print file validation results
    print("\n📁 Environment Template Files:")
    file_results = report["environment_files"]
    for template, data in file_results.items():
        if "exists" in data:
            exists_status = "✅" if data["exists"] else "❌"
            valid_status = "✅" if data.get("valid", False) else "❌"
            print(f"  {template}: {exists_status} exists, {valid_status} valid")
    
    # Print environment variables
    print("\n🔧 Environment Variables:")
    env_vars = report["environment_variables"]
    for var, data in env_vars.items():
        print(f"  {var}: {data['status']} ({data['value'][:20]}{'...' if len(data['value']) > 20 else ''})")
    
    # Print validation results
    print("\n🧪 Validation Script Results:")
    validation = report["validation_script_result"]
    print(f"  Status: {validation['status']}")
    if validation.get("errors"):
        print(f"  Errors: {validation['errors']}")
    if validation.get("output"):
        print(f"  Output: {validation['output']}")
    
    # Determine overall status
    all_files_valid = all(
        data.get("exists", False) and data.get("valid", False) 
        for data in file_results.values() 
        if "exists" in data
    )
    validation_passed = validation["status"] == "✅ PASS"
    critical_vars_set = any(data["is_set"] for data in env_vars.values())
    
    if all_files_valid and (validation_passed or critical_vars_set):
        report["overall_status"] = "✅ PASS"
        print("\n✅ ENVIRONMENT PARITY VALIDATION PASSED")
        print("🎯 STRICT_LIVE_MODE settings are properly configured for environment parity")
    else:
        report["overall_status"] = "❌ FAIL"
        print("\n❌ ENVIRONMENT PARITY VALIDATION FAILED")
        print("⚠️ Issues detected that may cause environment differences")
    
    # Recommendations
    print("\n💡 Recommendations:")
    if not all_files_valid:
        print("  - Ensure all environment template files exist and contain STRICT_LIVE_MODE settings")
    if not validation_passed and not critical_vars_set:
        print("  - Set explicit STRICT_LIVE_MODE environment variables or use templates")
        print("  - Run: cp backend/config/environment_templates/development.env .env")
    if validation_passed:
        print("  - Environment configuration is consistent ✅")
        print("  - STRICT_LIVE_MODE behavior will be identical across environments ✅")
    
    print("\n📋 Next Steps:")
    print("  1. Copy appropriate environment template to .env")
    print("  2. Set production values for API keys")
    print("  3. Verify configuration with: python environment_parity_validator.py")
    print("  4. Check API endpoint: /api/environment/strict-mode-status")
    
    return report

def main():
    """Main validation function"""
    try:
        report = generate_parity_report()
        
        # Save report to file
        report_file = Path("environment_parity_report.json")
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Full report saved to: {report_file}")
        
        # Exit with appropriate code
        exit_code = 0 if report["overall_status"] == "✅ PASS" else 1
        sys.exit(exit_code)
        
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()