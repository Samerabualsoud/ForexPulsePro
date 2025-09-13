"""
FastAPI Main Application
"""
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
from typing import List, Optional
from datetime import datetime

from .auth import verify_token, create_access_token
from .models import Signal, User, Strategy, NewsArticle, NewsSentiment
from .schemas import (
    SignalResponse, SignalCreate, UserCreate, StrategyUpdate, LoginRequest, 
    KillSwitchRequest, RiskConfigUpdate, NewsArticleResponse, NewsSentimentResponse,
    SentimentSummaryResponse, NewsAnalysisRequest, NewsFilters
)
from .database import get_db, SessionLocal
from .risk.guards import RiskManager
from .logs.logger import get_logger
from .services.signal_evaluator import evaluator
from .services.news_collector import news_collector
from sqlalchemy.orm import Session
from sqlalchemy import text
from prometheus_client import generate_latest
from fastapi.responses import Response
from .monitoring.metrics import metrics
from .api.monitoring import router as monitoring_router
import time
import asyncio
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    metrics.update_system_metrics()
    yield
    # Shutdown
    pass

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Forex Signal Dashboard API",
    description="Production-ready Forex Signal Dashboard REST API",
    version="1.0.0",
    lifespan=lifespan
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

# Security
security = HTTPBearer()
logger = get_logger(__name__)

# Lifespan function already defined above

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
    signal_text = f"{signal.symbol} {signal.action} @ {signal.price:.5f} | SL {signal.sl or 'N/A'} | TP {signal.tp or 'N/A'} | conf {signal.confidence:.2f}"
    
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
    token: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Toggle global kill switch (Admin only)"""
    user = verify_token(token.credentials)
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    risk_manager = RiskManager(db)
    risk_manager.set_kill_switch(request.enabled)
    
    logger.info(f"Kill switch {'enabled' if request.enabled else 'disabled'} by user {user.get('username')}")
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
    token: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Update risk configuration (Admin only)"""
    user = verify_token(token.credentials)
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    risk_manager = RiskManager(db)
    
    if config.daily_loss_limit is not None:
        risk_manager.set_daily_loss_limit(config.daily_loss_limit)
    if config.kill_switch_enabled is not None:
        risk_manager.set_kill_switch(config.kill_switch_enabled)
    
    logger.info(f"Risk config updated by user {user.get('username')}")
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
    token: HTTPAuthorizationCredentials = Depends(security),
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
    
    return Response(generate_latest(), media_type="text/plain")

@app.post("/api/auth/login")
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    """User authentication"""
    user = db.query(User).filter(User.username == request.username).first()
    if not user or not user.verify_password(request.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_access_token({"username": user.username, "role": user.role})
    return {"access_token": token, "token_type": "bearer", "role": user.role}

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
    token: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Evaluate expired signals and update their outcomes (Admin only)"""
    user = verify_token(token.credentials)
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
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
    token: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Simulate signal outcome for testing (Admin only)"""
    user = verify_token(token.credentials)
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    signal = db.query(Signal).filter(Signal.id == signal_id).first()
    if not signal:
        raise HTTPException(status_code=404, detail="Signal not found")
    
    if signal.result != "PENDING":
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
                id=s.id,
                news_article_id=s.news_article_id,
                analyzer_type=s.analyzer_type,
                sentiment_score=s.sentiment_score,
                confidence_score=s.confidence_score,
                sentiment_label=s.sentiment_label,
                analyzed_at=s.analyzed_at
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
    token: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Trigger news collection and sentiment analysis (Admin only)"""
    # Verify admin access
    user = verify_token(token.credentials)
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        # Run news collection
        summary = await news_collector.collect_all_news(
            force_refresh=request.force_refresh,
            symbols=request.symbols,
            categories=request.categories
        )
        
        logger.info(f"News analysis triggered by user {user.get('username')}: {summary['total_stored']} articles processed")
        
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
        symbols_covered = db.query(
            distinct(func.json_array_elements_text(NewsArticle.symbols))
        ).filter(
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
