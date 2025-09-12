"""
Pydantic Schemas
"""
from pydantic import BaseModel, validator
from datetime import datetime
from typing import Optional, Dict, Any

class SignalBase(BaseModel):
    symbol: str
    timeframe: str = "M1"
    action: str
    price: float
    sl: Optional[float] = None
    tp: Optional[float] = None
    confidence: float
    strategy: str
    version: str = "v1"
    expires_at: datetime

class SignalCreate(SignalBase):
    pass

class SignalResponse(SignalBase):
    id: int
    issued_at: datetime
    sent_to_whatsapp: bool
    blocked_by_risk: bool
    risk_reason: Optional[str] = None
    
    class Config:
        from_attributes = True

class UserCreate(BaseModel):
    username: str
    password: str
    role: str = "viewer"
    
    @validator('role')
    def validate_role(cls, v):
        if v not in ['admin', 'viewer']:
            raise ValueError('Role must be admin or viewer')
        return v

class UserResponse(BaseModel):
    id: int
    username: str
    role: str
    created_at: datetime
    is_active: bool
    
    class Config:
        from_attributes = True

class LoginRequest(BaseModel):
    username: str
    password: str

class KillSwitchRequest(BaseModel):
    enabled: bool

class StrategyUpdate(BaseModel):
    enabled: Optional[bool] = None
    config: Optional[Dict[str, Any]] = None

class StrategyResponse(BaseModel):
    id: int
    name: str
    symbol: str
    enabled: bool
    config: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class RiskConfigUpdate(BaseModel):
    kill_switch_enabled: Optional[bool] = None
    daily_loss_limit: Optional[float] = None
    volatility_guard_enabled: Optional[bool] = None
    volatility_threshold: Optional[float] = None
    max_daily_signals: Optional[int] = None

class LogEntryResponse(BaseModel):
    id: int
    timestamp: datetime
    level: str
    message: str
    source: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True
