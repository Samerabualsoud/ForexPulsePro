"""
Advanced Monitoring and Metrics API Endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session
from sqlalchemy import text, func
from datetime import datetime, timedelta
from typing import Dict, List, Any
import psutil
import time

from ..database import get_db
from ..models import Signal, Strategy, User
from ..auth import verify_token
from ..monitoring.metrics import metrics
from ..logs.logger import get_logger

router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])
security = HTTPBearer()
logger = get_logger(__name__)

@router.get("/dashboard")
async def get_monitoring_dashboard(
    token: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Get comprehensive monitoring dashboard data"""
    user = verify_token(token.credentials)
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Database metrics
        signal_count_24h = db.query(func.count(Signal.id)).filter(
            Signal.issued_at >= datetime.utcnow() - timedelta(hours=24)
        ).scalar()
        
        total_signals = db.query(func.count(Signal.id)).scalar()
        
        # Signals by strategy (last 24h)
        strategy_stats = db.query(
            Signal.strategy,
            func.count(Signal.id).label('count'),
            func.avg(Signal.confidence).label('avg_confidence')
        ).filter(
            Signal.issued_at >= datetime.utcnow() - timedelta(hours=24)
        ).group_by(Signal.strategy).all()
        
        # Signals by symbol (last 24h) 
        symbol_stats = db.query(
            Signal.symbol,
            func.count(Signal.id).label('count')
        ).filter(
            Signal.issued_at >= datetime.utcnow() - timedelta(hours=24)
        ).group_by(Signal.symbol).all()
        
        # Active strategies
        active_strategies = db.query(Strategy).filter(Strategy.enabled == True).count()
        total_strategies = db.query(Strategy).count()
        
        return {
            "system": {
                "cpu_usage_percent": cpu_percent,
                "memory_usage_percent": memory.percent,
                "memory_used_gb": round(memory.used / (1024**3), 2),
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "uptime_seconds": time.time() - metrics.app_start_time
            },
            "signals": {
                "total_signals": total_signals,
                "signals_24h": signal_count_24h,
                "strategies_active": active_strategies,
                "strategies_total": total_strategies,
                "by_strategy": [
                    {
                        "strategy": stat.strategy,
                        "count": stat.count,
                        "avg_confidence": round(float(stat.avg_confidence or 0), 3)
                    } for stat in strategy_stats
                ],
                "by_symbol": [
                    {
                        "symbol": stat.symbol,
                        "count": stat.count
                    } for stat in symbol_stats
                ]
            },
            "providers": {
                # This would be populated with real provider status
                "freecurrency": {"status": "available", "last_check": datetime.utcnow().isoformat()},
                "alphavantage": {"status": "available", "last_check": datetime.utcnow().isoformat()},
                "mock": {"status": "available", "last_check": datetime.utcnow().isoformat()}
            },
            "whatsapp": {
                "status": "configured",
                "last_message": None  # Would track actual WhatsApp stats
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get monitoring dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts")
async def get_system_alerts(
    token: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Get system alerts and warnings"""
    user = verify_token(token.credentials)
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    alerts = []
    
    try:
        # Check system resources
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            alerts.append({
                "type": "warning",
                "category": "system",
                "message": f"High memory usage: {memory.percent:.1f}%",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 85:
            alerts.append({
                "type": "warning", 
                "category": "system",
                "message": f"High CPU usage: {cpu_percent:.1f}%",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # Check recent signal generation
        recent_signals = db.query(func.count(Signal.id)).filter(
            Signal.issued_at >= datetime.utcnow() - timedelta(minutes=5)
        ).scalar()
        
        if recent_signals == 0:
            alerts.append({
                "type": "warning",
                "category": "signals",
                "message": "No signals generated in the last 5 minutes",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # Check database connectivity
        try:
            db.execute(text("SELECT 1"))
        except Exception:
            alerts.append({
                "type": "error",
                "category": "database", 
                "message": "Database connection failed",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # Check disabled strategies
        disabled_strategies = db.query(Strategy).filter(Strategy.enabled == False).count()
        if disabled_strategies > 0:
            alerts.append({
                "type": "info",
                "category": "strategies",
                "message": f"{disabled_strategies} strategies are disabled",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        return {"alerts": alerts, "alert_count": len(alerts)}
        
    except Exception as e:
        logger.error(f"Failed to get system alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance")
async def get_performance_metrics(
    token: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Get performance metrics and statistics"""
    user = verify_token(token.credentials)
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        # Signal performance over time
        hourly_signals = db.execute(text("""
            SELECT 
                DATE_TRUNC('hour', issued_at) as hour,
                COUNT(*) as signal_count,
                AVG(confidence) as avg_confidence
            FROM signals 
            WHERE issued_at >= NOW() - INTERVAL '24 hours'
            GROUP BY DATE_TRUNC('hour', issued_at)
            ORDER BY hour
        """)).fetchall()
        
        # Strategy performance
        strategy_performance = db.execute(text("""
            SELECT 
                strategy,
                COUNT(*) as total_signals,
                AVG(confidence) as avg_confidence,
                MIN(confidence) as min_confidence,
                MAX(confidence) as max_confidence
            FROM signals 
            WHERE issued_at >= NOW() - INTERVAL '24 hours'
            GROUP BY strategy
            ORDER BY total_signals DESC
        """)).fetchall()
        
        return {
            "hourly_signals": [
                {
                    "hour": row.hour.isoformat(),
                    "signal_count": row.signal_count,
                    "avg_confidence": round(float(row.avg_confidence or 0), 3)
                } for row in hourly_signals
            ],
            "strategy_performance": [
                {
                    "strategy": row.strategy,
                    "total_signals": row.total_signals,
                    "avg_confidence": round(float(row.avg_confidence or 0), 3),
                    "min_confidence": round(float(row.min_confidence or 0), 3),
                    "max_confidence": round(float(row.max_confidence or 0), 3)
                } for row in strategy_performance
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))