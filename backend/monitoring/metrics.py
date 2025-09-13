"""
Comprehensive Prometheus Metrics for Forex Signal Dashboard
"""
from prometheus_client import Counter, Histogram, Gauge, Info, Enum, CollectorRegistry, REGISTRY
import time
import psutil
from datetime import datetime
from typing import Dict, Any, Optional
from ..logs.logger import get_logger

logger = get_logger(__name__)

class PrometheusMetrics:
    """Central metrics collection for the Forex Signal Dashboard"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PrometheusMetrics, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        
        # ============================
        # SIGNAL GENERATION METRICS
        # ============================
        
        # Signal counts by strategy and symbol
        self.signals_generated_total = Counter(
            'forex_signals_generated_total',
            'Total signals generated',
            ['strategy', 'symbol', 'signal_type']
        )
        
        # Signal confidence distribution
        self.signal_confidence_histogram = Histogram(
            'forex_signal_confidence_score',
            'Distribution of signal confidence scores',
            ['strategy', 'symbol'],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        
        # Signal processing time by strategy
        self.signal_processing_time = Histogram(
            'forex_signal_processing_seconds',
            'Time spent processing signals by strategy',
            ['strategy', 'symbol'],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        
        # Risk management actions
        self.risk_actions_total = Counter(
            'forex_risk_actions_total',
            'Total risk management actions taken',
            ['action_type', 'reason']
        )
        
        # ============================
        # DATA PROVIDER METRICS
        # ============================
        
        # API calls by provider
        self.api_calls_total = Counter(
            'forex_api_calls_total',
            'Total API calls to data providers',
            ['provider', 'endpoint', 'status']
        )
        
        # API response time
        self.api_response_time = Histogram(
            'forex_api_response_seconds',
            'API response time by provider',
            ['provider', 'endpoint'],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]
        )
        
        # Provider availability
        self.provider_availability = Gauge(
            'forex_provider_availability',
            'Data provider availability status',
            ['provider']
        )
        
        # Rate limit status
        self.rate_limit_remaining = Gauge(
            'forex_rate_limit_remaining',
            'Remaining API calls before rate limit',
            ['provider']
        )
        
        # Fallback usage
        self.provider_fallback_total = Counter(
            'forex_provider_fallback_total',
            'Total fallback to alternative providers',
            ['from_provider', 'to_provider', 'reason']
        )
        
        # ============================
        # WHATSAPP DELIVERY METRICS
        # ============================
        
        # WhatsApp message delivery
        self.whatsapp_messages_total = Counter(
            'forex_whatsapp_messages_total',
            'Total WhatsApp messages by status',
            ['status', 'message_type']
        )
        
        # WhatsApp delivery time
        self.whatsapp_delivery_time = Histogram(
            'forex_whatsapp_delivery_seconds',
            'WhatsApp message delivery time',
            ['message_type'],
            buckets=[1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
        )
        
        # WhatsApp errors by type
        self.whatsapp_errors_total = Counter(
            'forex_whatsapp_errors_total',
            'WhatsApp errors by type',
            ['error_type', 'error_code']
        )
        
        # ============================
        # SYSTEM HEALTH METRICS
        # ============================
        
        # Application info
        self.app_info = Info(
            'forex_app_info',
            'Application information'
        )
        
        # Database connectivity
        self.database_connection_status = Gauge(
            'forex_database_connection_status',
            'Database connection status (1=connected, 0=disconnected)'
        )
        
        # Database query time
        self.database_query_time = Histogram(
            'forex_database_query_seconds',
            'Database query execution time',
            ['query_type'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
        )
        
        # System resource usage
        self.cpu_usage_percent = Gauge(
            'forex_cpu_usage_percent',
            'CPU usage percentage'
        )
        
        self.memory_usage_bytes = Gauge(
            'forex_memory_usage_bytes',
            'Memory usage in bytes'
        )
        
        self.memory_usage_percent = Gauge(
            'forex_memory_usage_percent',
            'Memory usage percentage'
        )
        
        # Active database connections
        self.database_connections_active = Gauge(
            'forex_database_connections_active',
            'Number of active database connections'
        )
        
        # Kill switch status
        self.kill_switch_status = Gauge(
            'forex_kill_switch_status',
            'Kill switch status (1=enabled, 0=disabled)'
        )
        
        # Application uptime
        self.app_uptime_seconds = Gauge(
            'forex_app_uptime_seconds',
            'Application uptime in seconds'
        )
        
        # ============================
        # BUSINESS METRICS
        # ============================
        
        # Strategy performance tracking
        self.strategy_performance = Gauge(
            'forex_strategy_performance_score',
            'Strategy performance score',
            ['strategy', 'timeframe']
        )
        
        # Market volatility
        self.market_volatility = Gauge(
            'forex_market_volatility',
            'Market volatility by symbol',
            ['symbol']
        )
        
        # Signal accuracy (when backtesting data available)
        self.signal_accuracy_percent = Gauge(
            'forex_signal_accuracy_percent',
            'Signal accuracy percentage',
            ['strategy', 'symbol', 'timeframe']
        )
        
        # Initialize app info
        self._initialize_app_info()
        
        # Track app start time
        self.app_start_time = time.time()
        
    def _initialize_app_info(self):
        """Initialize application information"""
        self.app_info.info({
            'version': '1.0.0',
            'name': 'forex_signal_dashboard',
            'description': 'Production Forex Signal Dashboard',
            'start_time': datetime.utcnow().isoformat()
        })
    
    def record_signal_generated(self, strategy: str, symbol: str, signal_type: str, 
                               confidence: float, processing_time: float):
        """Record signal generation metrics"""
        self.signals_generated_total.labels(
            strategy=strategy,
            symbol=symbol,
            signal_type=signal_type
        ).inc()
        
        self.signal_confidence_histogram.labels(
            strategy=strategy,
            symbol=symbol
        ).observe(confidence)
        
        self.signal_processing_time.labels(
            strategy=strategy,
            symbol=symbol
        ).observe(processing_time)
    
    def record_api_call(self, provider: str, endpoint: str, status: str, 
                       response_time: float):
        """Record API call metrics"""
        self.api_calls_total.labels(
            provider=provider,
            endpoint=endpoint,
            status=status
        ).inc()
        
        self.api_response_time.labels(
            provider=provider,
            endpoint=endpoint
        ).observe(response_time)
    
    def update_provider_availability(self, provider: str, is_available: bool):
        """Update provider availability status"""
        self.provider_availability.labels(provider=provider).set(1 if is_available else 0)
    
    def update_rate_limit(self, provider: str, remaining_calls: int):
        """Update rate limit remaining calls"""
        self.rate_limit_remaining.labels(provider=provider).set(remaining_calls)
    
    def record_provider_fallback(self, from_provider: str, to_provider: str, reason: str):
        """Record provider fallback event"""
        self.provider_fallback_total.labels(
            from_provider=from_provider,
            to_provider=to_provider,
            reason=reason
        ).inc()
    
    def record_whatsapp_message(self, status: str, message_type: str, 
                               delivery_time: Optional[float] = None):
        """Record WhatsApp message metrics"""
        self.whatsapp_messages_total.labels(
            status=status,
            message_type=message_type
        ).inc()
        
        if delivery_time is not None:
            self.whatsapp_delivery_time.labels(
                message_type=message_type
            ).observe(delivery_time)
    
    def record_whatsapp_error(self, error_type: str, error_code: str = "unknown"):
        """Record WhatsApp error"""
        self.whatsapp_errors_total.labels(
            error_type=error_type,
            error_code=error_code
        ).inc()
    
    def record_risk_action(self, action_type: str, reason: str):
        """Record risk management action"""
        self.risk_actions_total.labels(
            action_type=action_type,
            reason=reason
        ).inc()
    
    def record_database_query(self, query_type: str, execution_time: float):
        """Record database query metrics"""
        self.database_query_time.labels(query_type=query_type).observe(execution_time)
    
    def update_system_metrics(self):
        """Update system resource metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            self.cpu_usage_percent.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.memory_usage_bytes.set(memory.used)
            self.memory_usage_percent.set(memory.percent)
            
            # App uptime
            uptime = time.time() - self.app_start_time
            self.app_uptime_seconds.set(uptime)
            
        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")
    
    def update_database_status(self, is_connected: bool, active_connections: int = 0):
        """Update database status metrics"""
        self.database_connection_status.set(1 if is_connected else 0)
        if active_connections >= 0:
            self.database_connections_active.set(active_connections)
    
    def update_kill_switch(self, is_enabled: bool):
        """Update kill switch status"""
        self.kill_switch_status.set(1 if is_enabled else 0)
    
    def update_strategy_performance(self, strategy: str, timeframe: str, score: float):
        """Update strategy performance score"""
        self.strategy_performance.labels(
            strategy=strategy,
            timeframe=timeframe
        ).set(score)
    
    def update_market_volatility(self, symbol: str, volatility: float):
        """Update market volatility metric"""
        self.market_volatility.labels(symbol=symbol).set(volatility)
    
    def update_signal_accuracy(self, strategy: str, symbol: str, timeframe: str, accuracy: float):
        """Update signal accuracy percentage"""
        self.signal_accuracy_percent.labels(
            strategy=strategy,
            symbol=symbol,
            timeframe=timeframe
        ).set(accuracy)

# Global metrics instance
metrics = PrometheusMetrics()