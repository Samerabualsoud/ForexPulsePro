"""
Pydantic Schemas
"""
from pydantic import BaseModel, validator
from datetime import datetime
from typing import Optional, Dict, Any, List

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
    blocked_by_risk: bool
    risk_reason: Optional[str] = None
    
    # Success rate tracking fields
    tp_reached: Optional[bool] = None
    sl_hit: Optional[bool] = None
    result: str = "PENDING"
    evaluated_at: Optional[datetime] = None
    pips_result: Optional[float] = None
    
    # Immediate execution fields
    immediate_execution: bool = False
    urgency_level: str = "NORMAL"
    immediate_expiry: Optional[datetime] = None
    execution_window: int = 0
    
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

class NewsSentimentResponse(BaseModel):
    id: int
    news_article_id: int
    analyzer_type: str
    sentiment_score: float
    confidence_score: float
    sentiment_label: str
    analyzed_at: datetime
    
    class Config:
        from_attributes = True

class NewsArticleResponse(BaseModel):
    id: int
    title: str
    summary: Optional[str] = None
    content: Optional[str] = None
    url: str
    source: str
    published_at: datetime
    retrieved_at: datetime
    category: Optional[str] = None
    symbols: Optional[List[str]] = None
    is_relevant: bool
    sentiments: Optional[List[NewsSentimentResponse]] = None
    
    class Config:
        from_attributes = True

class SentimentSummaryResponse(BaseModel):
    overall_sentiment: str  # POSITIVE, NEGATIVE, NEUTRAL
    overall_score: float    # -1 to 1
    confidence: float       # 0 to 1
    total_articles: int
    positive_articles: int
    negative_articles: int
    neutral_articles: int
    timeframe: str          # e.g., "24h", "7d"
    by_symbol: Optional[Dict[str, Dict[str, Any]]] = None  # Symbol-specific sentiment
    by_source: Optional[Dict[str, Dict[str, Any]]] = None  # Source-specific sentiment

class NewsAnalysisRequest(BaseModel):
    force_refresh: bool = False
    symbols: Optional[List[str]] = None  # Specific symbols to analyze
    categories: Optional[List[str]] = None  # Specific categories
    
class NewsFilters(BaseModel):
    symbol: Optional[str] = None
    category: Optional[str] = None
    days: int = 7  # Look back days
    limit: int = 50
    include_sentiment: bool = True
    
    @validator('days')
    def validate_days(cls, v):
        if v < 1 or v > 30:
            raise ValueError('Days must be between 1 and 30')
        return v
        
    @validator('limit')
    def validate_limit(cls, v):
        if v < 1 or v > 1000:
            raise ValueError('Limit must be between 1 and 1000')
        return v
