"""
FastAPI Main Application
"""
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
from typing import List, Optional
from datetime import datetime

from .auth import verify_token, create_access_token
from .models import Signal, User, Strategy
from .schemas import SignalResponse, SignalCreate, UserCreate, StrategyUpdate, LoginRequest, KillSwitchRequest, RiskConfigUpdate
from .database import get_db, SessionLocal
from .risk.guards import RiskManager
from .logs.logger import get_logger
from .services.signal_evaluator import evaluator
from sqlalchemy.orm import Session
from sqlalchemy import text
from prometheus_client import generate_latest
from fastapi.responses import Response
from .monitoring.metrics import metrics
from .api.monitoring import router as monitoring_router
import time
import asyncio
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    metrics.update_system_metrics()
    yield
    # Shutdown
    pass

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Forex Signal Dashboard API",
    description="Production-ready Forex Signal Dashboard REST API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include monitoring router
app.include_router(monitoring_router)

# Security
security = HTTPBearer()
logger = get_logger(__name__)

# Lifespan function already defined above

@app.get("/api/health")
async def health_check(db: Session = Depends(get_db)):
    """Comprehensive health check endpoint"""
    # Update system metrics
    metrics.update_system_metrics()
    
    # Check database connectivity
    try:
        db.execute(text("SELECT 1"))
        db_healthy = True
        metrics.update_database_status(True)
    except Exception as e:
        db_healthy = False
        metrics.update_database_status(False)
        logger.error(f"Database health check failed: {e}")
    
    return {
        "status": "healthy" if db_healthy else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "database": "connected" if db_healthy else "disconnected",
        "services": {
            "signal_engine": "active",
            "database": "connected" if db_healthy else "disconnected",
            "whatsapp": "configured",  # Will be enhanced with actual status
            "monitoring": "active"
        }
    }

@app.get("/api/signals/latest")
async def get_latest_signal(
    symbol: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get latest signal for a symbol or all symbols"""
    query = db.query(Signal)
    if symbol:
        query = query.filter(Signal.symbol == symbol.upper())
    
    signal = query.order_by(Signal.issued_at.desc()).first()
    if not signal:
        raise HTTPException(status_code=404, detail="No signals found")
    
    return SignalResponse.from_orm(signal)

@app.get("/api/signals/recent")
async def get_recent_signals(
    limit: int = 50,
    symbol: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get recent signals"""
    query = db.query(Signal)
    if symbol:
        query = query.filter(Signal.symbol == symbol.upper())
    
    signals = query.order_by(Signal.issued_at.desc()).limit(limit).all()
    return [SignalResponse.from_orm(signal) for signal in signals]


@app.post("/api/risk/killswitch")
async def toggle_killswitch(
    request: KillSwitchRequest,
    token: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Toggle global kill switch (Admin only)"""
    user = verify_token(token.credentials)
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    risk_manager = RiskManager(db)
    risk_manager.set_kill_switch(request.enabled)
    
    logger.info(f"Kill switch {'enabled' if request.enabled else 'disabled'} by user {user.get('username')}")
    return {"status": "success", "kill_switch_enabled": request.enabled}

@app.get("/api/risk/status")
async def get_risk_status(db: Session = Depends(get_db)):
    """Get current risk management status"""
    risk_manager = RiskManager(db)
    return {
        "kill_switch_enabled": risk_manager.is_kill_switch_active(),
        "daily_loss_limit": risk_manager.get_daily_loss_limit(),
        "current_daily_loss": risk_manager.get_current_daily_loss(),
        "volatility_guard_enabled": risk_manager.is_volatility_guard_active()
    }

@app.put("/api/risk/config")
async def update_risk_config(
    config: RiskConfigUpdate,
    token: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Update risk configuration (Admin only)"""
    user = verify_token(token.credentials)
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    risk_manager = RiskManager(db)
    
    if config.daily_loss_limit is not None:
        risk_manager.set_daily_loss_limit(config.daily_loss_limit)
    if config.kill_switch_enabled is not None:
        risk_manager.set_kill_switch(config.kill_switch_enabled)
    
    logger.info(f"Risk config updated by user {user.get('username')}")
    return {"status": "success", "config": config}

@app.get("/api/strategies")
async def get_strategies(db: Session = Depends(get_db)):
    """Get all strategy configurations"""
    strategies = db.query(Strategy).all()
    return strategies

@app.put("/api/strategies/{strategy_id}")
async def update_strategy(
    strategy_id: int,
    strategy_update: StrategyUpdate,
    token: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Update strategy configuration (Admin only)"""
    user = verify_token(token.credentials)
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    strategy = db.query(Strategy).filter(Strategy.id == strategy_id).first()
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    for field, value in strategy_update.dict(exclude_unset=True).items():
        setattr(strategy, field, value)
    
    db.commit()
    db.refresh(strategy)
    
    logger.info(f"Strategy {strategy_id} updated by user {user.get('username')}")
    return strategy

@app.get("/metrics")
async def get_metrics(db: Session = Depends(get_db)):
    """Enhanced Prometheus metrics endpoint"""
    # Update all system metrics before serving
    metrics.update_system_metrics()
    
    # Check and update database status
    try:
        db.execute(text("SELECT 1"))
        metrics.update_database_status(True)
    except Exception:
        metrics.update_database_status(False)
    
    return Response(generate_latest(), media_type="text/plain")

@app.post("/api/auth/login")
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    """User authentication"""
    user = db.query(User).filter(User.username == request.username).first()
    if not user or not user.verify_password(request.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_access_token({"username": user.username, "role": user.role})
    return {"access_token": token, "token_type": "bearer", "role": user.role}

@app.get("/api/signals/success-rate")
async def get_success_rate(
    days: int = 30,
    db: Session = Depends(get_db)
):
    """Get signal success rate statistics"""
    try:
        stats = evaluator.get_success_rate_stats(db, days)
        return stats
    except Exception as e:
        logger.error(f"Error getting success rate stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get success rate statistics")

@app.post("/api/signals/evaluate-expired")
async def evaluate_expired_signals(
    token: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Evaluate expired signals and update their outcomes (Admin only)"""
    user = verify_token(token.credentials)
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        results = evaluator.evaluate_expired_signals(db)
        return {
            "status": "success",
            "message": f"Evaluated {results['evaluated_count']} expired signals",
            "results": results
        }
    except Exception as e:
        logger.error(f"Error evaluating expired signals: {e}")
        raise HTTPException(status_code=500, detail="Failed to evaluate expired signals")

@app.post("/api/signals/{signal_id}/simulate")
async def simulate_signal_outcome(
    signal_id: int,
    token: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Simulate signal outcome for testing (Admin only)"""
    user = verify_token(token.credentials)
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    signal = db.query(Signal).filter(Signal.id == signal_id).first()
    if not signal:
        raise HTTPException(status_code=404, detail="Signal not found")
    
    if signal.result != "PENDING":
        raise HTTPException(status_code=400, detail="Signal already evaluated")
    
    try:
        evaluator.simulate_signal_outcome(signal, db)
        return {"status": "success", "message": f"Signal {signal_id} outcome simulated"}
    except Exception as e:
        logger.error(f"Error simulating signal outcome: {e}")
        raise HTTPException(status_code=500, detail="Failed to simulate signal outcome")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
