"""
Database Configuration and Initialization
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from .models import Base, User, Strategy, RiskConfig
from .logs.logger import get_logger

logger = get_logger(__name__)

# Database URL with fallback to SQLite
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    # Fallback to SQLite
    DATABASE_URL = "sqlite:///./forex_signals.db"
    logger.info("Using SQLite database (fallback)")
else:
    logger.info("Using PostgreSQL database")

# Create engine
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
else:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db() -> Session:
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize database tables"""
    try:
        Base.metadata.create_all(bind=engine, checkfirst=True)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        # Don't raise if tables already exist
        if "already exists" not in str(e):
            raise

def create_default_data():
    """Create default users, strategies, and configurations"""
    db = SessionLocal()
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
        
        # Create default strategies
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
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
                    'min_confidence': 0.6,
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
                    'min_confidence': 0.65,
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
