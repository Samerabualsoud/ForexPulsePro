"""
FastAPI Main Application
"""
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import List, Optional, TYPE_CHECKING
from datetime import datetime

# Authentication removed - no longer needed
from .models import Signal, User, Strategy, NewsArticle, NewsSentiment
from .schemas import (
    SignalResponse, SignalCreate, UserCreate, StrategyUpdate, LoginRequest, 
    KillSwitchRequest, RiskConfigUpdate, NewsArticleResponse, NewsSentimentResponse,
    SentimentSummaryResponse, NewsAnalysisRequest, NewsFilters
)
from .database import get_db, get_session_local, get_db_fingerprint, dispose_engine
from .risk.guards import RiskManager
from .logs.logger import get_logger
from .services.signal_evaluator import evaluator
from .services.news_collector import news_collector
from .services.provider_diagnostics import provider_diagnostics_service
from sqlalchemy.orm import Session
from sqlalchemy import text
from prometheus_client import generate_latest
from fastapi.responses import Response
from .monitoring.metrics import metrics
from .api.monitoring import router as monitoring_router
from .api.environment_validation import router as environment_router
import time
import asyncio
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    metrics.update_system_metrics()
    
    # Initialize AI agents for proper cleanup tracking
    ai_agents = []
    signal_scheduler = None
    
    try:
        from .services.multi_ai_consensus import MultiAIConsensus
        consensus_system = MultiAIConsensus()
        # DeepSeek agent disabled per user request
        if consensus_system.groq_agent:
            ai_agents.append(consensus_system.groq_agent)
        logger.info(f"Initialized {len(ai_agents)} AI agents for resource management")
    except Exception as e:
        logger.warning(f"Could not initialize AI agents for cleanup: {e}")
    
    # **CRITICAL FIX**: Start the signal generation scheduler with proper cleanup
    try:
        from .scheduler import SignalScheduler
        signal_scheduler = SignalScheduler()
        signal_scheduler.start()
        logger.info("Signal generation scheduler initialized and started")
    except Exception as e:
        logger.error(f"Failed to start signal scheduler: {e}")
    
    try:
        yield
    finally:
        # Shutdown - cleanup scheduler first to prevent interpreter shutdown errors
        if signal_scheduler:
            try:
                logger.info("Shutting down signal scheduler...")
                signal_scheduler.scheduler.shutdown(wait=False)
                logger.info("Signal scheduler shutdown complete")
            except Exception as e:
                logger.error(f"Error shutting down scheduler: {e}")
        
        # Shutdown - cleanup AI agent resources
        logger.info("Starting AI agent cleanup...")
        for agent in ai_agents:
            try:
                if hasattr(agent, 'cleanup'):
                    await agent.cleanup()
                    logger.info(f"Cleaned up {agent.__class__.__name__}")
            except Exception as e:
                logger.error(f"Error cleaning up {agent.__class__.__name__}: {e}")
        
        # Dispose database engine to clean up connection pools
        try:
            dispose_engine()
            logger.info("Database engine disposed successfully")
        except Exception as e:
            logger.error(f"Error disposing database engine: {e}")
        
        logger.info("Shutdown complete")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Forex Signal Dashboard API",
    description="Production-ready Forex Signal Dashboard REST API",
    version="1.0.0",
    lifespan=lifespan  # type: ignore
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
app.include_router(environment_router)

logger = get_logger(__name__)

# Lifespan function already defined above

@app.get("/api/db_fingerprint")
async def database_fingerprint():
    """Get database connection fingerprint for diagnostics"""
    try:
        fingerprint = get_db_fingerprint()
        return {
            "status": "success",
            "fingerprint": fingerprint,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get database fingerprint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get database fingerprint: {str(e)}"
        )

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

@app.get("/api/diagnostics/providers")
async def get_provider_diagnostics():
    """
    Comprehensive provider diagnostics endpoint for configuration verification.
    
    This endpoint provides detailed information about:
    - Provider configurations and status
    - Strict live mode settings
    - Environment detection
    - Configuration fingerprint for environment comparison
    - Health checks and troubleshooting information
    
    Use the configuration_fingerprint to verify identical configurations
    between development and production environments.
    """
    try:
        diagnostics = provider_diagnostics_service.get_comprehensive_diagnostics()
        logger.info(f"Provider diagnostics generated successfully: {diagnostics['configuration_fingerprint']}")
        return diagnostics
    except Exception as e:
        logger.error(f"Failed to generate provider diagnostics: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate provider diagnostics: {str(e)}"
        )

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

@app.post("/api/signals/{signal_id}/test")
async def test_signal(
    signal_id: int,
    db: Session = Depends(get_db)
):
    """Test signal formatting and validation"""
    signal = db.query(Signal).filter(Signal.id == signal_id).first()
    if not signal:
        raise HTTPException(status_code=404, detail="Signal not found")
    
    # Format signal text for testing
    signal_text = f"{signal.symbol} {signal.action} @ {signal.price:.5f} | SL {signal.sl if signal.sl is not None else 'N/A'} | TP {signal.tp if signal.tp is not None else 'N/A'} | conf {signal.confidence:.2f}"
    
    logger.info(f"Signal {signal_id} test message generated: {signal_text}")
    
    return {
        "status": "success",
        "signal_id": signal_id,
        "test_message": signal_text,
        "formatted_at": datetime.utcnow().isoformat()
    }

@app.post("/api/signals/{signal_id}/resend")
async def resend_signal(
    signal_id: int,
    db: Session = Depends(get_db)
):
    """Resend signal (placeholder for WhatsApp integration)"""
    signal = db.query(Signal).filter(Signal.id == signal_id).first()
    if not signal:
        raise HTTPException(status_code=404, detail="Signal not found")
    
    logger.info(f"Signal {signal_id} resend requested for {signal.symbol} {signal.action}")
    
    # In a real implementation, this would trigger WhatsApp/Telegram send
    # For now, we'll just mark it as a successful operation
    
    return {
        "status": "success",
        "signal_id": signal_id,
        "message": f"Signal {signal_id} marked for resend",
        "resent_at": datetime.utcnow().isoformat()
    }


@app.post("/api/risk/killswitch")
async def toggle_killswitch(
    request: KillSwitchRequest,
    db: Session = Depends(get_db)
):
    """Toggle global kill switch (No auth required)"""
    
    risk_manager = RiskManager(db)
    risk_manager.set_kill_switch(request.enabled)
    
    logger.info(f"Kill switch {'enabled' if request.enabled else 'disabled'}")
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
    db: Session = Depends(get_db)
):
    """Update risk configuration (No auth required)"""
    
    risk_manager = RiskManager(db)
    
    if config.daily_loss_limit is not None:
        risk_manager.set_daily_loss_limit(config.daily_loss_limit)
    if config.kill_switch_enabled is not None:
        risk_manager.set_kill_switch(config.kill_switch_enabled)
    
    logger.info(f"Risk config updated")
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
    db: Session = Depends(get_db)
):
    """Update strategy configuration (No auth required)"""
    
    strategy = db.query(Strategy).filter(Strategy.id == strategy_id).first()
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    for field, value in strategy_update.dict(exclude_unset=True).items():
        setattr(strategy, field, value)
    
    db.commit()
    db.refresh(strategy)
    
    logger.info(f"Strategy {strategy_id} updated")
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
    
    return Response(generate_latest(registry=metrics.registry), media_type="text/plain")

# Login endpoint removed - no authentication required

@app.get("/api/signals/stats")
async def get_signal_stats(db: Session = Depends(get_db)):
    """Get basic signal statistics"""
    try:
        total_signals = db.query(Signal).count()
        active_signals = db.query(Signal).filter(Signal.result == None).count()
        winning_signals = db.query(Signal).filter(Signal.result == 'WIN').count()
        losing_signals = db.query(Signal).filter(Signal.result == 'LOSS').count()
        
        return {
            "total": total_signals,
            "active": active_signals,
            "winning": winning_signals,
            "losing": losing_signals,
            "win_rate": (winning_signals / (winning_signals + losing_signals) * 100) if (winning_signals + losing_signals) > 0 else 0
        }
    except Exception as e:
        logger.error(f"Error getting signal stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get signal statistics")

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
    db: Session = Depends(get_db)
):
    """Evaluate expired signals and update their outcomes (No auth required)"""
    
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
    db: Session = Depends(get_db)
):
    """Simulate signal outcome for testing (No auth required)"""
    
    signal = db.query(Signal).filter(Signal.id == signal_id).first()
    if not signal:
        raise HTTPException(status_code=404, detail="Signal not found")
    
    if signal.result != "PENDING":  # type: ignore
        raise HTTPException(status_code=400, detail="Signal already evaluated")
    
    try:
        evaluator.simulate_signal_outcome(signal, db)
        return {"status": "success", "message": f"Signal {signal_id} outcome simulated"}
    except Exception as e:
        logger.error(f"Error simulating signal outcome: {e}")
        raise HTTPException(status_code=500, detail="Failed to simulate signal outcome")

# News and Sentiment API Endpoints

@app.get("/api/news/feed", response_model=List[NewsArticleResponse])
async def get_news_feed(
    filters: NewsFilters = Depends(),
    db: Session = Depends(get_db)
):
    """Get latest news articles with optional filtering"""
    try:
        articles = news_collector.get_news_articles(
            db=db,
            symbol=filters.symbol,
            category=filters.category,
            days=filters.days,
            limit=filters.limit,
            include_sentiment=filters.include_sentiment
        )
        
        # Convert to response format
        response_articles = []
        for article in articles:
            article_dict = {
                'id': article.id,
                'title': article.title,
                'summary': article.summary,
                'content': article.content,
                'url': article.url,
                'source': article.source,
                'published_at': article.published_at,
                'retrieved_at': article.retrieved_at,
                'category': article.category,
                'symbols': article.symbols,
                'is_relevant': article.is_relevant,
                'sentiments': [
                    {
                        'id': s.id,
                        'news_article_id': s.news_article_id,
                        'analyzer_type': s.analyzer_type,
                        'sentiment_score': s.sentiment_score,
                        'confidence_score': s.confidence_score,
                        'sentiment_label': s.sentiment_label,
                        'analyzed_at': s.analyzed_at
                    } for s in article.sentiments
                ] if filters.include_sentiment and article.sentiments else None
            }
            response_articles.append(NewsArticleResponse(**article_dict))
        
        logger.info(f"Retrieved {len(response_articles)} news articles with filters: {filters.dict()}")
        return response_articles
        
    except Exception as e:
        logger.error(f"Error retrieving news feed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve news feed")

@app.get("/api/news/sentiment/{article_id}", response_model=List[NewsSentimentResponse])
async def get_article_sentiment(
    article_id: int,
    db: Session = Depends(get_db)
):
    """Get sentiment analysis for a specific news article"""
    try:
        # Check if article exists
        article = db.query(NewsArticle).filter(NewsArticle.id == article_id).first()
        if not article:
            raise HTTPException(status_code=404, detail="News article not found")
        
        # Get sentiment data
        sentiments = db.query(NewsSentiment).filter(
            NewsSentiment.news_article_id == article_id
        ).all()
        
        if not sentiments:
            # Trigger sentiment analysis if not present
            success = await news_collector._analyze_article_sentiment(article)
            if success:
                # Refetch sentiments
                sentiments = db.query(NewsSentiment).filter(
                    NewsSentiment.news_article_id == article_id
                ).all()
        
        response_sentiments = [
            NewsSentimentResponse(
                id=s.id,  # type: ignore
                news_article_id=s.news_article_id,  # type: ignore
                analyzer_type=s.analyzer_type,  # type: ignore
                sentiment_score=s.sentiment_score,  # type: ignore
                confidence_score=s.confidence_score,  # type: ignore
                sentiment_label=s.sentiment_label,  # type: ignore
                analyzed_at=s.analyzed_at  # type: ignore
            ) for s in sentiments
        ]
        
        logger.info(f"Retrieved {len(response_sentiments)} sentiment analyses for article {article_id}")
        return response_sentiments
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving sentiment for article {article_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve sentiment analysis")

@app.get("/api/news/sentiment-summary", response_model=SentimentSummaryResponse)
async def get_sentiment_summary(
    symbol: Optional[str] = None,
    days: int = 7,
    db: Session = Depends(get_db)
):
    """Get overall market sentiment summary for a timeframe"""
    try:
        # Validate days parameter
        if days < 1 or days > 30:
            raise HTTPException(status_code=400, detail="Days parameter must be between 1 and 30")
        
        summary = news_collector.get_sentiment_summary(
            db=db,
            symbol=symbol,
            days=days
        )
        
        logger.info(f"Generated sentiment summary for symbol={symbol}, days={days}")
        return SentimentSummaryResponse(**summary)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating sentiment summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate sentiment summary")

@app.post("/api/news/analyze")
async def trigger_news_analysis(
    request: NewsAnalysisRequest,
    db: Session = Depends(get_db)
):
    """Trigger news collection and sentiment analysis (No auth required)"""
    
    try:
        # Run news collection
        summary = await news_collector.collect_all_news(
            force_refresh=request.force_refresh,
            symbols=request.symbols,
            categories=request.categories
        )
        
        logger.info(f"News analysis triggered: {summary['total_stored']} articles processed")
        
        return {
            "status": "success",
            "message": f"News collection completed: {summary['total_stored']} articles stored, {summary['total_analyzed']} analyzed",
            "summary": summary
        }
        
    except Exception as e:
        logger.error(f"Error in news analysis: {e}")
        raise HTTPException(status_code=500, detail="Failed to complete news analysis")

@app.get("/api/news/providers/status")
async def get_news_providers_status():
    """Get status of all news data providers"""
    try:
        alphavantage_status = {
            "name": "AlphaVantage",
            "enabled": news_collector.alphavantage.enabled,
            "api_key_configured": bool(news_collector.alphavantage.api_key),
            "rate_limit_calls": len(news_collector.alphavantage.call_timestamps),
            "capabilities": ["news", "sentiment", "symbol_news"]
        }
        
        finnhub_status = {
            "name": "Finnhub", 
            "enabled": news_collector.finnhub.is_available(),
            "api_key_configured": bool(news_collector.finnhub.api_key),
            "capabilities": ["news", "market_news", "symbol_news"]
        }
        
        return {
            "providers": [alphavantage_status, finnhub_status],
            "total_enabled": sum(1 for p in [alphavantage_status, finnhub_status] if p["enabled"])
        }
        
    except Exception as e:
        logger.error(f"Error getting provider status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get provider status")

@app.get("/api/system/production-mode")
async def get_production_mode_status():
    """Check if system is in production mode (using live data) vs demo mode"""
    try:
        from .config.provider_config import deterministic_provider_config
        
        # Initialize providers to check their availability
        try:
            from .providers.polygon_provider import PolygonProvider
            from .providers.freecurrency import FreeCurrencyAPIProvider
            from .providers.finnhub_provider import FinnhubProvider
            from .providers.coinbase_provider import CoinbaseProvider
            
            # Check live data providers
            live_providers = []
            
            # Check major live providers
            polygon = PolygonProvider()
            if polygon.is_available():
                live_providers.append({"name": "Polygon.io", "type": "live_real_time", "status": "available"})
            
            freecurrency = FreeCurrencyAPIProvider()
            if hasattr(freecurrency, 'api_key') and freecurrency.api_key:
                live_providers.append({"name": "FreeCurrencyAPI", "type": "live_real_time", "status": "available"})
            elif not hasattr(freecurrency, '_is_api_available') or freecurrency._is_api_available != False:
                # FreeCurrencyAPI works without API key for basic usage
                live_providers.append({"name": "FreeCurrencyAPI", "type": "live_real_time", "status": "available"})
            
            finnhub = FinnhubProvider()
            if finnhub.api_key and finnhub.client:
                live_providers.append({"name": "Finnhub", "type": "live_delayed", "status": "available"})
            
            coinbase = CoinbaseProvider()
            # Only add Coinbase if it's actually available and can be verified
            try:
                if hasattr(coinbase, 'is_available') and coinbase.is_available():
                    live_providers.append({"name": "Coinbase", "type": "live_real_time", "status": "available"})
                else:
                    logger.debug("Coinbase provider not available - failed is_available() check")
            except Exception as coinbase_error:
                logger.debug(f"Coinbase provider check failed: {coinbase_error}")
            
            # Determine production mode
            is_production = len(live_providers) > 0
            
            return {
                "is_production_mode": is_production,
                "live_providers_count": len(live_providers),
                "active_live_providers": live_providers,
                "status": "🟢 LIVE DATA" if is_production else "🟡 DEMO MODE",
                "data_source": "live" if is_production else "demo",
                "last_checked": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"Error checking provider status: {e}")
            # Fallback: assume production if we have any API keys configured
            has_polygon = bool(os.getenv('POLYGON_API_KEY'))
            has_finnhub = bool(os.getenv('FINNHUB_API_KEY'))
            has_freecurrency = bool(os.getenv('FREECURRENCY_API_KEY'))
            
            has_live_providers = has_polygon or has_finnhub or has_freecurrency
            
            return {
                "is_production_mode": has_live_providers,
                "live_providers_count": sum([has_polygon, has_finnhub, has_freecurrency]),
                "active_live_providers": [],
                "status": "🟢 LIVE DATA" if has_live_providers else "🟡 DEMO MODE",
                "data_source": "live" if has_live_providers else "demo", 
                "last_checked": datetime.utcnow().isoformat(),
                "note": "Provider status check failed, using API key detection"
            }
            
    except Exception as e:
        logger.error(f"Error getting production mode status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get production mode status")

@app.get("/api/news/stats")
async def get_news_statistics(
    days: int = 7,
    db: Session = Depends(get_db)
):
    """Get news collection and sentiment statistics"""
    try:
        from datetime import timedelta
        from sqlalchemy import func, distinct
        
        # Validate days parameter
        if days < 1 or days > 90:
            raise HTTPException(status_code=400, detail="Days parameter must be between 1 and 90")
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Article statistics
        total_articles = db.query(NewsArticle).filter(
            NewsArticle.retrieved_at >= cutoff_date
        ).count()
        
        articles_by_source = db.query(
            NewsArticle.source,
            func.count(NewsArticle.id).label('count')
        ).filter(
            NewsArticle.retrieved_at >= cutoff_date
        ).group_by(NewsArticle.source).all()
        
        articles_by_category = db.query(
            NewsArticle.category,
            func.count(NewsArticle.id).label('count')
        ).filter(
            NewsArticle.retrieved_at >= cutoff_date
        ).group_by(NewsArticle.category).all()
        
        # Sentiment statistics
        total_sentiments = db.query(NewsSentiment).join(NewsArticle).filter(
            NewsArticle.retrieved_at >= cutoff_date
        ).count()
        
        sentiment_by_label = db.query(
            NewsSentiment.sentiment_label,
            func.count(NewsSentiment.id).label('count')
        ).join(NewsArticle).filter(
            NewsArticle.retrieved_at >= cutoff_date
        ).group_by(NewsSentiment.sentiment_label).all()
        
        # Average sentiment scores
        avg_sentiment_score = db.query(
            func.avg(NewsSentiment.sentiment_score)
        ).join(NewsArticle).filter(
            NewsArticle.retrieved_at >= cutoff_date
        ).scalar() or 0.0
        
        # Unique symbols covered  
        symbols_covered = db.query(NewsArticle).filter(
            NewsArticle.retrieved_at >= cutoff_date,
            NewsArticle.symbols.isnot(None)
        ).count()
        
        return {
            "timeframe_days": days,
            "articles": {
                "total": total_articles,
                "by_source": {source: count for source, count in articles_by_source},
                "by_category": {category: count for category, count in articles_by_category}
            },
            "sentiment": {
                "total_analyzed": total_sentiments,
                "average_score": round(float(avg_sentiment_score), 4),
                "by_label": {label: count for label, count in sentiment_by_label}
            },
            "coverage": {
                "unique_symbols": symbols_covered
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting news statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get news statistics")

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Use environment variable for port configuration
    port = int(os.getenv("BACKEND_PORT", "8000"))
    host = os.getenv("BACKEND_HOST", "0.0.0.0")
    
    uvicorn.run(app, host=host, port=port)
