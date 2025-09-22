"""
Database Configuration and Initialization
"""
import os
import threading
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator, Optional
from sqlalchemy.pool import StaticPool, NullPool

from .models import Base, User, Strategy, RiskConfig
from .logs.logger import get_logger

logger = get_logger(__name__)

def get_database_url():
    """Get database URL dynamically from environment"""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        # Fallback to SQLite
        database_url = "sqlite:///./forex_signals.db"
        logger.info("Using SQLite database (fallback)")
    else:
        # Log only host info to avoid credential leakage
        try:
            from urllib.parse import urlparse
            parsed = urlparse(database_url)
            logger.info(f"Using PostgreSQL database: {parsed.hostname}:{parsed.port}/{parsed.path.lstrip('/')}")
        except:
            logger.info("Using PostgreSQL database: [host info unavailable]")
    return database_url

class DatabaseEngineManager:
    """Thread-safe database engine manager that can hot-swap engines"""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._engine = None
        self._sessionmaker = None
        self._current_url = None
        self._instance_id = 0
    
    def get_engine(self):
        """Get current engine, creating new one if URL changed"""
        current_url = get_database_url()
        
        with self._lock:
            if self._engine is None or self._current_url != current_url:
                # Dispose old engine if exists
                if self._engine is not None:
                    logger.info(f"Disposing old database engine (URL changed)")
                    self._engine.dispose()
                
                # Create new engine
                if current_url.startswith("sqlite"):
                    self._engine = create_engine(
                        current_url,
                        connect_args={"check_same_thread": False},
                        poolclass=StaticPool
                    )
                else:
                    # Use NullPool temporarily to avoid connection caching issues
                    self._engine = create_engine(
                        current_url, 
                        pool_pre_ping=True,
                        poolclass=NullPool
                    )
                
                self._current_url = current_url
                self._sessionmaker = None  # Reset sessionmaker
                self._instance_id += 1
                logger.info(f"Created new database engine (instance #{self._instance_id})")
            
            return self._engine
    
    def get_sessionmaker(self):
        """Get current sessionmaker, creating new one if engine changed"""
        engine = self.get_engine()  # This handles URL changes
        
        with self._lock:
            if self._sessionmaker is None:
                self._sessionmaker = sessionmaker(autocommit=False, autoflush=False, bind=engine)
                logger.info(f"Created new sessionmaker for engine instance #{self._instance_id}")
            
            return self._sessionmaker
    
    def dispose(self):
        """Dispose of current engine and reset"""
        with self._lock:
            if self._engine is not None:
                self._engine.dispose()
                self._engine = None
                self._sessionmaker = None
                self._current_url = None
                logger.info("Database engine disposed")
    
    def get_fingerprint(self):
        """Get current database fingerprint for diagnostics"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(self._current_url or "")
            return {
                "host": parsed.hostname,
                "port": parsed.port,
                "database": parsed.path.lstrip('/') if parsed.path else None,
                "engine_instance_id": self._instance_id,
                "url_set": self._current_url is not None
            }
        except:
            return {
                "error": "Unable to parse URL",
                "engine_instance_id": self._instance_id,
                "url_set": self._current_url is not None
            }

# Global engine manager instance
_engine_manager = DatabaseEngineManager()

def get_engine():
    """Get current database engine"""
    return _engine_manager.get_engine()

def get_session_local():
    """Get current sessionmaker (hot-swappable when URL changes)"""
    return _engine_manager.get_sessionmaker()

def dispose_engine():
    """Dispose current engine (for cleanup)"""
    _engine_manager.dispose()

def get_db_fingerprint():
    """Get database connection fingerprint for diagnostics"""
    return _engine_manager.get_fingerprint()

def get_db() -> Generator[Session, None, None]:
    """Get database session with fresh sessionmaker"""
    SessionLocal = get_session_local()  # Always get fresh sessionmaker
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize database tables"""
    try:
        current_engine = get_engine()
        Base.metadata.create_all(bind=current_engine, checkfirst=True)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        # Don't raise if tables already exist
        if "already exists" not in str(e):
            raise

def create_default_data():
    """Create default users, strategies, and configurations"""
    session_local = get_session_local()
    db = session_local()
    try:
        # Create default admin user if not exists
        admin_user = db.query(User).filter(User.username == "admin").first()
        if not admin_user:
            admin_user = User(username="admin", role="admin")
            admin_user.set_password("admin123")  # Change in production
            db.add(admin_user)
            logger.info("Created default admin user")
        
        # Create default viewer user if not exists
        viewer_user = db.query(User).filter(User.username == "viewer").first()
        if not viewer_user:
            viewer_user = User(username="viewer", role="viewer")
            viewer_user.set_password("viewer123")  # Change in production
            db.add(viewer_user)
            logger.info("Created default viewer user")
        
        # Create default strategies for forex and commodities
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD', 'XAGUSD', 'USOIL']
        strategies = [
            {
                'name': 'ema_rsi',
                'config': {
                    'ema_fast': 12,
                    'ema_slow': 26,
                    'rsi_period': 14,
                    'rsi_buy_threshold': 50,
                    'rsi_sell_threshold': 50,
                    'sl_mode': 'atr',
                    'sl_multiplier': 2.0,
                    'tp_mode': 'atr',
                    'tp_multiplier': 3.0,
                    'min_confidence': 0.60,  # 60% minimum for forex majors
                    'expiry_bars': 60
                }
            },
            {
                'name': 'donchian_atr',
                'config': {
                    'donchian_period': 20,
                    'atr_period': 14,
                    'atr_multiplier': 2.0,
                    'use_supertrend': True,
                    'sl_mode': 'atr',
                    'sl_multiplier': 2.0,
                    'tp_mode': 'atr',
                    'tp_multiplier': 3.0,
                    'min_confidence': 0.67,  # 67% minimum for crypto (raised from 65%)
                    'expiry_bars': 45
                }
            },
            {
                'name': 'meanrev_bb',
                'config': {
                    'bb_period': 20,
                    'bb_std': 2.0,
                    'adx_period': 14,
                    'adx_threshold': 25,
                    'zscore_threshold': 2.0,
                    'sl_mode': 'pips',
                    'sl_pips': 20,
                    'tp_mode': 'pips',
                    'tp_pips': 40,
                    'min_confidence': 0.7,
                    'expiry_bars': 30
                }
            }
        ]
        
        for symbol in symbols:
            for strategy_config in strategies:
                existing = db.query(Strategy).filter(
                    Strategy.name == strategy_config['name'],
                    Strategy.symbol == symbol
                ).first()
                
                if not existing:
                    strategy = Strategy(
                        name=strategy_config['name'],
                        symbol=symbol,
                        enabled=True,
                        config=strategy_config['config']
                    )
                    db.add(strategy)
        
        # Create default risk configuration
        risk_config = db.query(RiskConfig).first()
        if not risk_config:
            risk_config = RiskConfig(
                kill_switch_enabled=False,
                daily_loss_limit=1000.0,
                volatility_guard_enabled=True,
                volatility_threshold=0.02,
                max_daily_signals=100
            )
            db.add(risk_config)
            logger.info("Created default risk configuration")
        
        db.commit()
        logger.info("Default data created successfully")
        
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create default data: {e}")
        raise
    finally:
        db.close()
