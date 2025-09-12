"""
Structured Logging Configuration
"""
import structlog
import logging
import logging.handlers
import os
import json
from datetime import datetime
from pathlib import Path

# Create logs directory
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

def get_logger(name: str = None):
    """Get configured structured logger"""
    return structlog.get_logger(name)

def setup_logging():
    """Setup structured logging with rotation"""
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=None,
        level=logging.INFO,
    )
    
    # Setup file handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        logs_dir / "forex_signals.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    
    # Setup error file handler
    error_handler = logging.handlers.RotatingFileHandler(
        logs_dir / "forex_signals_errors.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    
    # Add handlers to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    
    # Setup application logger
    app_logger = logging.getLogger("forex_signals")
    app_logger.setLevel(logging.INFO)

class DatabaseLogHandler(logging.Handler):
    """Custom log handler to store logs in database"""
    
    def __init__(self, db_session_factory):
        super().__init__()
        self.db_session_factory = db_session_factory
    
    def emit(self, record):
        """Emit log record to database"""
        try:
            from ..models import LogEntry
            
            db = self.db_session_factory()
            
            # Parse structured log data
            log_data = {}
            if hasattr(record, 'msg') and isinstance(record.msg, dict):
                log_data = record.msg
            
            log_entry = LogEntry(
                level=record.levelname,
                message=record.getMessage(),
                source=record.name,
                data=log_data
            )
            
            db.add(log_entry)
            db.commit()
            db.close()
            
        except Exception as e:
            # Don't let logging errors break the application
            print(f"Database logging error: {e}")

def log_signal_generated(signal_data: dict, strategy: str):
    """Log signal generation event"""
    logger = get_logger("signal_engine")
    logger.info(
        "Signal generated",
        signal=signal_data,
        strategy=strategy,
        event_type="signal_generated"
    )

def log_whatsapp_sent(signal_id: int, recipients: int, success: bool):
    """Log WhatsApp message event"""
    logger = get_logger("whatsapp_service")
    logger.info(
        "WhatsApp message sent",
        signal_id=signal_id,
        recipients=recipients,
        success=success,
        event_type="whatsapp_sent"
    )

def log_risk_block(signal_data: dict, reason: str):
    """Log risk management block event"""
    logger = get_logger("risk_manager")
    logger.warning(
        "Signal blocked by risk management",
        signal=signal_data,
        reason=reason,
        event_type="risk_block"
    )

def log_api_request(endpoint: str, method: str, user: str = None, success: bool = True):
    """Log API request event"""
    logger = get_logger("api")
    logger.info(
        "API request",
        endpoint=endpoint,
        method=method,
        user=user,
        success=success,
        event_type="api_request"
    )

def log_system_event(event: str, details: dict = None):
    """Log system event"""
    logger = get_logger("system")
    logger.info(
        event,
        details=details or {},
        event_type="system_event"
    )

# Initialize logging on import
setup_logging()
