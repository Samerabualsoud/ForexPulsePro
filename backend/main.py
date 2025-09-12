"""
FastAPI Main Application
"""
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
import os
from typing import List, Optional
from datetime import datetime

from .auth import verify_token, create_access_token
from .models import Signal, User, Strategy
from .schemas import SignalResponse, SignalCreate, UserCreate, StrategyUpdate
from .database import get_db, SessionLocal
from .services.whatsapp import WhatsAppService
from .risk.guards import RiskManager
from .logs.logger import get_logger
from sqlalchemy.orm import Session
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response

# Initialize FastAPI app
app = FastAPI(
    title="Forex Signal Dashboard API",
    description="Production-ready Forex Signal Dashboard REST API",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
logger = get_logger(__name__)

# Prometheus metrics
signals_generated_total = Counter('signals_generated_total', 'Total signals generated')
whatsapp_send_total = Counter('whatsapp_send_total', 'Total WhatsApp messages sent')
whatsapp_errors_total = Counter('whatsapp_errors_total', 'Total WhatsApp errors')
signal_processing_time = Histogram('signal_processing_seconds', 'Time spent processing signals')

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
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

@app.post("/api/signals/resend")
async def resend_signal(
    signal_id: int,
    token: str = Depends(security),
    db: Session = Depends(get_db)
):
    """Resend a signal to WhatsApp (Admin only)"""
    user = verify_token(token.credentials)
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    signal = db.query(Signal).filter(Signal.id == signal_id).first()
    if not signal:
        raise HTTPException(status_code=404, detail="Signal not found")
    
    whatsapp_service = WhatsAppService()
    try:
        result = await whatsapp_service.send_signal(signal)
        whatsapp_send_total.inc()
        logger.info(f"Signal {signal_id} resent successfully")
        return {"status": "sent", "message_id": result.get("message_id")}
    except Exception as e:
        whatsapp_errors_total.inc()
        logger.error(f"Failed to resend signal {signal_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/whatsapp/test")
async def test_whatsapp(
    token: str = Depends(security),
    db: Session = Depends(get_db)
):
    """Test WhatsApp connectivity (Admin only)"""
    user = verify_token(token.credentials)
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    whatsapp_service = WhatsAppService()
    try:
        result = await whatsapp_service.send_test_message()
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"WhatsApp test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/risk/killswitch")
async def toggle_killswitch(
    enabled: bool,
    token: str = Depends(security),
    db: Session = Depends(get_db)
):
    """Toggle global kill switch (Admin only)"""
    user = verify_token(token.credentials)
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    risk_manager = RiskManager(db)
    risk_manager.set_kill_switch(enabled)
    
    logger.info(f"Kill switch {'enabled' if enabled else 'disabled'} by user {user.get('username')}")
    return {"status": "success", "kill_switch_enabled": enabled}

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

@app.get("/api/strategies")
async def get_strategies(db: Session = Depends(get_db)):
    """Get all strategy configurations"""
    strategies = db.query(Strategy).all()
    return strategies

@app.put("/api/strategies/{strategy_id}")
async def update_strategy(
    strategy_id: int,
    strategy_update: StrategyUpdate,
    token: str = Depends(security),
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
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")

@app.post("/api/auth/login")
async def login(username: str, password: str, db: Session = Depends(get_db)):
    """User authentication"""
    user = db.query(User).filter(User.username == username).first()
    if not user or not user.verify_password(password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_access_token({"username": user.username, "role": user.role})
    return {"access_token": token, "token_type": "bearer", "role": user.role}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
