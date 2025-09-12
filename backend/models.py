"""
SQLAlchemy Models
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import hashlib
import os

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), default="viewer")  # admin, viewer
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    def set_password(self, password: str):
        """Hash and set password"""
        salt = os.urandom(32)
        self.password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000).hex() + salt.hex()
    
    def verify_password(self, password: str) -> bool:
        """Verify password"""
        if len(self.password_hash) < 64:
            return False
        hash_part = self.password_hash[:-64]
        salt_part = bytes.fromhex(self.password_hash[-64:])
        return hashlib.pbkdf2_hmac('sha256', password.encode(), salt_part, 100000).hex() == hash_part

class Signal(Base):
    __tablename__ = "signals"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), index=True, nullable=False)
    timeframe = Column(String(10), default="M1")
    action = Column(String(10), nullable=False)  # BUY, SELL, FLAT
    price = Column(Float, nullable=False)
    sl = Column(Float)  # Stop Loss
    tp = Column(Float)  # Take Profit
    confidence = Column(Float, nullable=False)
    strategy = Column(String(50), nullable=False)
    version = Column(String(10), default="v1")
    expires_at = Column(DateTime, nullable=False)
    issued_at = Column(DateTime, default=datetime.utcnow, index=True)
    sent_to_whatsapp = Column(Boolean, default=False)
    blocked_by_risk = Column(Boolean, default=False)
    risk_reason = Column(Text)
    
    # Signal outcome tracking
    tp_reached = Column(Boolean, default=None)  # True if take profit was hit
    sl_hit = Column(Boolean, default=None)      # True if stop loss was hit  
    result = Column(String(20), default="PENDING")  # PENDING, WIN, LOSS, EXPIRED
    evaluated_at = Column(DateTime, default=None)   # When outcome was determined
    pips_result = Column(Float, default=None)       # Actual pips gained/lost
    
    def to_dict(self):
        return {
            "id": self.id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "action": self.action,
            "price": self.price,
            "sl": self.sl,
            "tp": self.tp,
            "confidence": self.confidence,
            "strategy": self.strategy,
            "version": self.version,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "issued_at": self.issued_at.isoformat() if self.issued_at else None,
            "sent_to_whatsapp": self.sent_to_whatsapp,
            "blocked_by_risk": self.blocked_by_risk,
            "tp_reached": self.tp_reached,
            "sl_hit": self.sl_hit,
            "result": self.result,
            "evaluated_at": self.evaluated_at.isoformat() if self.evaluated_at else None,
            "pips_result": self.pips_result
        }

class Strategy(Base):
    __tablename__ = "strategies"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), nullable=False)
    symbol = Column(String(10), nullable=False)
    enabled = Column(Boolean, default=True)
    config = Column(JSON, nullable=False)  # Strategy-specific configuration
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (UniqueConstraint('name', 'symbol', name='strategies_name_symbol_key'),)

class RiskConfig(Base):
    __tablename__ = "risk_config"
    
    id = Column(Integer, primary_key=True, index=True)
    kill_switch_enabled = Column(Boolean, default=False)
    daily_loss_limit = Column(Float, default=1000.0)
    volatility_guard_enabled = Column(Boolean, default=True)
    volatility_threshold = Column(Float, default=0.02)  # 2% ATR threshold
    max_daily_signals = Column(Integer, default=100)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class LogEntry(Base):
    __tablename__ = "log_entries"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    level = Column(String(10), nullable=False)
    message = Column(Text, nullable=False)
    source = Column(String(50))
    data = Column(JSON)  # Additional structured data
