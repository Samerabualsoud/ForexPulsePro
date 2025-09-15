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
    action = Column(String(20), nullable=False)  # BUY, SELL, BUY LIMIT, SELL LIMIT, BUY STOP, SELL STOP, BUY STOP LIMIT, SELL STOP LIMIT, FLAT
    price = Column(Float, nullable=False)
    sl = Column(Float)  # Stop Loss
    tp = Column(Float)  # Take Profit
    confidence = Column(Float, nullable=False)
    strategy = Column(String(50), nullable=False)
    version = Column(String(10), default="v1")
    expires_at = Column(DateTime, nullable=False)
    issued_at = Column(DateTime, default=datetime.utcnow, index=True)
    blocked_by_risk = Column(Boolean, default=False)
    risk_reason = Column(Text)
    
    # Auto-trading fields
    auto_traded = Column(Boolean, default=False)
    auto_trade_failed = Column(Boolean, default=False)
    broker_ticket = Column(String, nullable=True)  # MT5 ticket number
    executed_price = Column(Float, nullable=True)  # Actual execution price
    executed_volume = Column(Float, nullable=True)  # Executed lot size
    execution_slippage = Column(Float, nullable=True)  # Slippage in pips
    execution_time = Column(DateTime, nullable=True)  # When trade was executed
    execution_error = Column(String, nullable=True)  # Error message if failed
    
    # Signal outcome tracking
    tp_reached = Column(Boolean, default=None)  # True if take profit was hit
    sl_hit = Column(Boolean, default=None)      # True if stop loss was hit  
    result = Column(String(20), default="PENDING")  # PENDING, WIN, LOSS, EXPIRED
    evaluated_at = Column(DateTime, default=None)   # When outcome was determined
    pips_result = Column(Float, default=None)       # Actual pips gained/lost
    
    # AI Consensus Enhancement Fields
    ai_consensus_confidence = Column(Float, nullable=True)  # Overall AI consensus confidence
    consensus_level = Column(String(30), nullable=True)    # HIGH_AGREEMENT, MODERATE_AGREEMENT, DISAGREEMENT
    ai_reasoning = Column(Text, nullable=True)             # AI reasoning for strategy selection
    manus_ai_confidence = Column(Float, nullable=True)     # Individual Manus AI confidence
    chatgpt_confidence = Column(Float, nullable=True)      # Individual ChatGPT confidence
    ai_enhanced = Column(Boolean, default=False)          # Whether signal was AI-enhanced
    strategy_ranking = Column(Integer, nullable=True)      # Strategy rank in AI consensus (1=top)
    conflict_resolution = Column(String(50), nullable=True) # How AI conflicts were resolved
    
    # Immediate Execution Fields
    immediate_execution = Column(Boolean, default=False)    # Whether signal needs immediate execution
    urgency_level = Column(String(20), default="NORMAL")   # NORMAL, HIGH, CRITICAL
    immediate_expiry = Column(DateTime, nullable=True)     # Quick expiry for immediate signals (5-15 minutes)
    execution_window = Column(Integer, default=0)          # Minutes for immediate execution window
    
    # Sentiment analysis fields removed temporarily to fix schema mismatch
    
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
            "blocked_by_risk": self.blocked_by_risk,
            "tp_reached": self.tp_reached,
            "sl_hit": self.sl_hit,
            "result": self.result,
            "evaluated_at": self.evaluated_at.isoformat() if self.evaluated_at else None,
            "pips_result": self.pips_result,
            "auto_traded": self.auto_traded,
            "broker_ticket": self.broker_ticket,
            "executed_price": self.executed_price,
            "executed_volume": self.executed_volume,
            "execution_slippage": self.execution_slippage,
            "execution_time": self.execution_time.isoformat() if self.execution_time else None,
            # AI Consensus fields
            "ai_consensus_confidence": self.ai_consensus_confidence,
            "consensus_level": self.consensus_level,
            "ai_reasoning": self.ai_reasoning,
            "manus_ai_confidence": self.manus_ai_confidence,
            "chatgpt_confidence": self.chatgpt_confidence,
            "ai_enhanced": self.ai_enhanced,
            "strategy_ranking": self.strategy_ranking,
            "conflict_resolution": self.conflict_resolution,
            # Immediate execution fields
            "immediate_execution": self.immediate_execution,
            "urgency_level": self.urgency_level,
            "immediate_expiry": self.immediate_expiry.isoformat() if self.immediate_expiry else None,
            "execution_window": self.execution_window
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

class AIConsensusHistory(Base):
    """Track AI consensus recommendations over time"""
    __tablename__ = "ai_consensus_history"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), index=True, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Consensus results
    consensus_level = Column(String(30), nullable=False)  # HIGH_AGREEMENT, MODERATE_AGREEMENT, DISAGREEMENT
    overall_confidence = Column(Float, nullable=False)
    agreement_score = Column(Float, nullable=False)
    
    # AI contributions
    manus_ai_data = Column(JSON, nullable=True)    # Manus AI full response
    chatgpt_data = Column(JSON, nullable=True)     # ChatGPT full response
    
    # Strategy recommendations
    recommended_strategies = Column(JSON, nullable=False)  # List of recommended strategies with rankings
    conflict_areas = Column(JSON, nullable=True)          # Areas where AIs disagreed
    reasoning = Column(Text, nullable=True)               # Consensus reasoning
    
    # Market context
    market_regime = Column(String(30), nullable=True)     # Market regime when consensus was made
    volatility_level = Column(String(20), nullable=True)  # Market volatility context
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            "id": self.id,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "consensus_level": self.consensus_level,
            "overall_confidence": self.overall_confidence,
            "agreement_score": self.agreement_score,
            "recommended_strategies": self.recommended_strategies,
            "conflict_areas": self.conflict_areas,
            "reasoning": self.reasoning,
            "market_regime": self.market_regime,
            "volatility_level": self.volatility_level,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class StrategyPerformance(Base):
    """Track individual strategy performance metrics"""
    __tablename__ = "strategy_performance"
    
    id = Column(Integer, primary_key=True, index=True)
    strategy_name = Column(String(50), nullable=False, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    evaluation_period = Column(String(20), nullable=False)  # daily, weekly, monthly
    
    # Performance metrics
    total_signals = Column(Integer, default=0)
    successful_signals = Column(Integer, default=0)
    win_rate = Column(Float, default=0.0)
    avg_confidence = Column(Float, default=0.0)
    total_pips = Column(Float, default=0.0)
    avg_pips_per_signal = Column(Float, default=0.0)
    
    # Risk metrics
    max_drawdown = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    profit_factor = Column(Float, default=0.0)
    
    # AI enhancement metrics
    ai_enhanced_signals = Column(Integer, default=0)
    ai_enhancement_improvement = Column(Float, default=0.0)  # Performance improvement with AI
    manus_ai_agreement_rate = Column(Float, default=0.0)
    chatgpt_agreement_rate = Column(Float, default=0.0)
    
    # Time period
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    calculated_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (UniqueConstraint('strategy_name', 'symbol', 'evaluation_period', 'period_start', 
                                       name='strategy_performance_unique'),)
    
    def to_dict(self):
        return {
            "id": self.id,
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "evaluation_period": self.evaluation_period,
            "total_signals": self.total_signals,
            "successful_signals": self.successful_signals,
            "win_rate": self.win_rate,
            "avg_confidence": self.avg_confidence,
            "total_pips": self.total_pips,
            "avg_pips_per_signal": self.avg_pips_per_signal,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "profit_factor": self.profit_factor,
            "ai_enhanced_signals": self.ai_enhanced_signals,
            "ai_enhancement_improvement": self.ai_enhancement_improvement,
            "manus_ai_agreement_rate": self.manus_ai_agreement_rate,
            "chatgpt_agreement_rate": self.chatgpt_agreement_rate,
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "calculated_at": self.calculated_at.isoformat() if self.calculated_at else None
        }

class BacktestResult(Base):
    """Store comprehensive backtesting results"""
    __tablename__ = "backtest_results"
    
    id = Column(Integer, primary_key=True, index=True)
    strategy_name = Column(String(50), nullable=False, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    backtest_type = Column(String(30), nullable=False)  # simple, monte_carlo, walk_forward, ai_enhanced
    
    # Test configuration
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    initial_capital = Column(Float, nullable=False)
    configuration = Column(JSON, nullable=True)  # Strategy parameters used
    
    # Core performance metrics
    total_return = Column(Float, default=0.0)
    annual_return = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    sortino_ratio = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    max_drawdown_duration = Column(Integer, default=0)
    
    # Trading metrics
    total_trades = Column(Integer, default=0)
    win_rate = Column(Float, default=0.0)
    profit_factor = Column(Float, default=0.0)
    avg_trade_return = Column(Float, default=0.0)
    volatility = Column(Float, default=0.0)
    
    # Risk metrics
    calmar_ratio = Column(Float, default=0.0)
    value_at_risk = Column(Float, default=0.0)
    conditional_var = Column(Float, default=0.0)
    
    # AI-specific metrics (for ai_enhanced backtests)
    ai_consensus_confidence = Column(Float, nullable=True)
    strategy_consistency_score = Column(Float, nullable=True)
    regime_adaptability = Column(Float, nullable=True)
    manus_ai_performance = Column(Float, nullable=True)
    chatgpt_performance = Column(Float, nullable=True)
    
    # Monte Carlo specific (for monte_carlo backtests)
    monte_carlo_runs = Column(Integer, nullable=True)
    worst_case_scenario = Column(Float, nullable=True)
    best_case_scenario = Column(Float, nullable=True)
    confidence_intervals = Column(JSON, nullable=True)
    
    # Walk-forward specific (for walk_forward backtests)
    walk_forward_periods = Column(Integer, nullable=True)
    out_of_sample_performance = Column(JSON, nullable=True)
    
    # Overall assessment
    overall_score = Column(Float, default=0.0)      # Composite score 0-100
    recommendation = Column(String(50), nullable=True)  # Strategy recommendation
    risk_assessment = Column(JSON, nullable=True)   # Risk analysis
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    execution_time_seconds = Column(Float, nullable=True)
    
    __table_args__ = (UniqueConstraint('strategy_name', 'symbol', 'backtest_type', 'start_date', 'end_date',
                                       name='backtest_results_unique'),)
    
    def to_dict(self):
        return {
            "id": self.id,
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "backtest_type": self.backtest_type,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "initial_capital": self.initial_capital,
            "configuration": self.configuration,
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_trade_return": self.avg_trade_return,
            "volatility": self.volatility,
            "calmar_ratio": self.calmar_ratio,
            "value_at_risk": self.value_at_risk,
            "conditional_var": self.conditional_var,
            "ai_consensus_confidence": self.ai_consensus_confidence,
            "strategy_consistency_score": self.strategy_consistency_score,
            "regime_adaptability": self.regime_adaptability,
            "overall_score": self.overall_score,
            "recommendation": self.recommendation,
            "risk_assessment": self.risk_assessment,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "execution_time_seconds": self.execution_time_seconds
        }

class AIStrategyOptimization(Base):
    """Track AI-driven strategy parameter optimization history"""
    __tablename__ = "ai_strategy_optimization"
    
    id = Column(Integer, primary_key=True, index=True)
    strategy_name = Column(String(50), nullable=False, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    optimization_type = Column(String(30), nullable=False)  # parameter_sweep, ai_guided, monte_carlo
    
    # Optimization configuration
    optimization_metric = Column(String(30), nullable=False)  # sharpe_ratio, total_return, calmar_ratio
    parameter_ranges = Column(JSON, nullable=False)          # Parameters that were optimized
    
    # Results
    best_parameters = Column(JSON, nullable=False)
    best_performance = Column(JSON, nullable=False)
    optimization_iterations = Column(Integer, default=0)
    improvement_percentage = Column(Float, default=0.0)
    
    # AI validation
    manus_ai_validation = Column(JSON, nullable=True)
    chatgpt_validation = Column(JSON, nullable=True)
    ai_consensus_approval = Column(Boolean, default=False)
    ai_confidence = Column(Float, nullable=True)
    
    # Market context
    market_conditions = Column(JSON, nullable=True)
    optimization_period = Column(JSON, nullable=False)  # start/end dates
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    optimization_time_seconds = Column(Float, nullable=True)
    status = Column(String(20), default="COMPLETED")  # RUNNING, COMPLETED, FAILED
    
    def to_dict(self):
        return {
            "id": self.id,
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "optimization_type": self.optimization_type,
            "optimization_metric": self.optimization_metric,
            "parameter_ranges": self.parameter_ranges,
            "best_parameters": self.best_parameters,
            "best_performance": self.best_performance,
            "optimization_iterations": self.optimization_iterations,
            "improvement_percentage": self.improvement_percentage,
            "ai_consensus_approval": self.ai_consensus_approval,
            "ai_confidence": self.ai_confidence,
            "market_conditions": self.market_conditions,
            "optimization_period": self.optimization_period,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "optimization_time_seconds": self.optimization_time_seconds,
            "status": self.status
        }

class MarketRegime(Base):
    __tablename__ = "market_regimes"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), index=True, nullable=False)
    regime = Column(String(20), nullable=False)  # TRENDING, RANGING, HIGH_VOLATILITY, STRONG_TRENDING
    confidence = Column(Float, nullable=False)
    adx = Column(Float)
    atr_ratio = Column(Float)
    detected_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Index for efficient queries
    __table_args__ = (
        UniqueConstraint('symbol', 'detected_at', name='_symbol_time_uc'),
    )

class LogEntry(Base):
    __tablename__ = "log_entries"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    level = Column(String(10), nullable=False)
    message = Column(Text, nullable=False)
    source = Column(String(50))
    data = Column(JSON)  # Additional structured data

class NewsArticle(Base):
    __tablename__ = "news_articles"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(500), nullable=False, index=True)
    summary = Column(Text)
    content = Column(Text)
    url = Column(String(1000), unique=True, nullable=False, index=True)
    source = Column(String(100), nullable=False, index=True)
    published_at = Column(DateTime, nullable=False, index=True)
    retrieved_at = Column(DateTime, default=datetime.utcnow, index=True)
    category = Column(String(50), index=True)  # 'forex', 'crypto', 'general'
    symbols = Column(JSON)  # Related trading symbols ['EURUSD', 'GBPUSD']
    is_relevant = Column(Boolean, default=True, index=True)
    
    # Relationship to sentiment analysis
    sentiments = relationship("NewsSentiment", back_populates="article", cascade="all, delete-orphan")
    
    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "summary": self.summary,
            "content": self.content,
            "url": self.url,
            "source": self.source,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "retrieved_at": self.retrieved_at.isoformat() if self.retrieved_at else None,
            "category": self.category,
            "symbols": self.symbols,
            "is_relevant": self.is_relevant
        }

class NewsSentiment(Base):
    __tablename__ = "news_sentiments"
    
    id = Column(Integer, primary_key=True, index=True)
    news_article_id = Column(Integer, ForeignKey("news_articles.id"), nullable=False, index=True)
    analyzer_type = Column(String(50), nullable=False, index=True)  # 'vader', 'textblob', 'financial_keywords', 'combined'
    sentiment_score = Column(Float, nullable=False)  # -1 to 1 range
    confidence_score = Column(Float, nullable=False)  # 0 to 1 range
    sentiment_label = Column(String(20), nullable=False, index=True)  # 'POSITIVE', 'NEGATIVE', 'NEUTRAL'
    analyzed_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationship back to news article
    article = relationship("NewsArticle", back_populates="sentiments")
    
    # Unique constraint to prevent duplicate analysis for same article/analyzer
    __table_args__ = (
        UniqueConstraint('news_article_id', 'analyzer_type', name='_article_analyzer_uc'),
    )
    
    def to_dict(self):
        return {
            "id": self.id,
            "news_article_id": self.news_article_id,
            "analyzer_type": self.analyzer_type,
            "sentiment_score": self.sentiment_score,
            "confidence_score": self.confidence_score,
            "sentiment_label": self.sentiment_label,
            "analyzed_at": self.analyzed_at.isoformat() if self.analyzed_at else None
        }
