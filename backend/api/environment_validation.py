"""
Environment Configuration Validation API Endpoints
Runtime validation of STRICT_LIVE_MODE settings and provider configuration
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from datetime import datetime
import os

from ..database import get_db
from ..config.strict_live_config import StrictLiveConfig
from ..config.provider_validation import get_provider_validation_service
from ..logs.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/environment", tags=["Environment Validation"])

@router.get("/strict-mode-status")
async def get_strict_mode_status() -> Dict[str, Any]:
    """Get current STRICT_LIVE_MODE configuration and status"""
    try:
        status = StrictLiveConfig.get_status_summary()
        
        # Add runtime validation
        validation_service = get_provider_validation_service()
        validation_result = validation_service.validate_environment_configuration()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "strict_mode_configuration": status,
            "validation_result": {
                "configuration_fingerprint": validation_result["configuration_fingerprint"],
                "validation_passed": validation_result["validation_passed"],
                "total_issues": len(validation_result["issues"]),
                "total_warnings": len(validation_result["warnings"]),
                "environment": validation_result["environment"]
            },
            "provider_status": validation_result["provider_status"]
        }
    except Exception as e:
        logger.error(f"Failed to get strict mode status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@router.get("/configuration-fingerprint")
async def get_configuration_fingerprint() -> Dict[str, str]:
    """Get environment configuration fingerprint for comparison"""
    try:
        validation_service = get_provider_validation_service()
        fingerprint = validation_service.generate_configuration_fingerprint()
        
        return {
            "configuration_fingerprint": fingerprint,
            "environment": os.getenv("ENVIRONMENT", "unknown"),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to generate configuration fingerprint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate fingerprint: {str(e)}")

@router.post("/validate-environment-parity")
async def validate_environment_parity(reference_fingerprint: str) -> Dict[str, Any]:
    """Validate current environment against a reference configuration fingerprint"""
    try:
        validation_service = get_provider_validation_service()
        comparison = validation_service.compare_with_reference_configuration(reference_fingerprint)
        
        return {
            "parity_check": comparison,
            "timestamp": datetime.utcnow().isoformat(),
            "recommendation": "Configuration matches reference" if comparison["matches_reference"] 
                           else "Configuration mismatch detected - review environment variables"
        }
    except Exception as e:
        logger.error(f"Failed to validate environment parity: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to validate parity: {str(e)}")

@router.get("/configuration-report")
async def get_configuration_report() -> Dict[str, Any]:
    """Get comprehensive configuration report for troubleshooting"""
    try:
        validation_service = get_provider_validation_service()
        report = validation_service.generate_configuration_report()
        
        return {
            "report": report,
            "generated_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to generate configuration report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")

@router.get("/environment-variables")
async def get_environment_variables() -> Dict[str, Any]:
    """Get masked environment variables for debugging (API keys masked)"""
    try:
        validation_service = get_provider_validation_service()
        validation_result = validation_service.validate_environment_configuration()
        
        # Add additional environment context
        environment_info = {
            "environment_variables": validation_result["environment_variables"],
            "strict_mode_enabled": StrictLiveConfig.ENABLED,
            "approved_sources": StrictLiveConfig.APPROVED_LIVE_SOURCES,
            "blocked_sources": StrictLiveConfig.BLOCKED_SOURCES,
            "max_data_age": StrictLiveConfig.MAX_DATA_AGE_SECONDS,
            "min_providers": StrictLiveConfig.MIN_PROVIDER_VALIDATIONS
        }
        
        return {
            "environment_info": environment_info,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get environment variables: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get environment variables: {str(e)}")

@router.get("/deployment-checklist")
async def get_deployment_checklist() -> Dict[str, Any]:
    """Get deployment readiness checklist for environment parity"""
    try:
        validation_service = get_provider_validation_service()
        validation_result = validation_service.validate_environment_configuration()
        
        checklist = {
            "strict_mode_configuration": {
                "status": "✅ PASS" if StrictLiveConfig.ENABLED else "⚠️ DISABLED",
                "details": f"STRICT_LIVE_MODE={StrictLiveConfig.ENABLED}"
            },
            "provider_validation": {
                "status": "✅ PASS" if validation_result["validation_passed"] else "❌ FAIL",
                "details": f"{len(validation_result['issues'])} issues, {len(validation_result['warnings'])} warnings"
            },
            "api_keys_configured": {
                "status": "✅ PASS" if all(
                    os.getenv(key, '').strip() != '' 
                    for key in ['POLYGON_API_KEY', 'FINNHUB_API_KEY', 'FREECURRENCY_API_KEY']
                ) else "⚠️ PARTIAL",
                "details": "Core API keys configured"
            },
            "configuration_fingerprint": validation_result["configuration_fingerprint"],
            "ready_for_production": validation_result["validation_passed"] and StrictLiveConfig.ENABLED
        }
        
        return {
            "deployment_checklist": checklist,
            "overall_status": "✅ READY" if checklist["ready_for_production"] else "❌ NOT READY",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to generate deployment checklist: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate checklist: {str(e)}")

@router.get("/docs")
async def get_environment_documentation() -> Dict[str, str]:
    """Get environment configuration documentation"""
    return {
        "environment_variables_documentation": StrictLiveConfig.get_environment_variables_doc(),
        "validation_script_usage": """
# Environment Validation Script Usage

## Validate current environment against file:
python -m backend.config.environment_validation --current --env1 backend/config/environment_templates/production.env

## Compare two environment files:
python -m backend.config.environment_validation --env1 development.env --env2 production.env

## Generate report to file:
python -m backend.config.environment_validation --current --env1 production.env --output validation_report.txt

## Quiet mode (only critical issues):
python -m backend.config.environment_validation --current --env1 production.env --quiet
        """,
        "deployment_process": """
# Environment Parity Deployment Process

1. Copy environment template:
   cp backend/config/environment_templates/production.env .env

2. Set production values:
   - Replace all "your_*_here" placeholders with real credentials
   - Ensure STRICT_LIVE_MODE settings match development

3. Validate configuration:
   python -m backend.config.environment_validation --current --env1 backend/config/environment_templates/development.env

4. Check API endpoint:
   curl http://your-domain/api/environment/strict-mode-status

5. Verify fingerprints match:
   curl http://your-domain/api/environment/configuration-fingerprint
        """
    }