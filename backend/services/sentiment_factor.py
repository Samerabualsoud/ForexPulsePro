"""
Sentiment Factor Service

This service calculates sentiment impact on trading signals by analyzing
recent news data and sentiment scores for specific currency pairs and symbols.
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func
import os

from ..models import NewsArticle, NewsSentiment
from ..logs.logger import get_logger
from ..config.sentiment_config import sentiment_config

logger = get_logger(__name__)


class SentimentFactorService:
    """
    Service to calculate sentiment impact on trading signals
    
    Analyzes recent news sentiment to provide bias factors that can
    enhance or reduce signal confidence based on market sentiment.
    """
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        
        # Configuration from centralized config
        self.sentiment_enabled = sentiment_config.ENABLED
        self.sentiment_weight = sentiment_config.WEIGHT
        self.lookback_hours = sentiment_config.LOOKBACK_HOURS
        self.recency_decay = sentiment_config.RECENCY_DECAY
        
        # Sentiment thresholds
        self.positive_threshold = sentiment_config.POSITIVE_THRESHOLD
        self.negative_threshold = sentiment_config.NEGATIVE_THRESHOLD
        self.high_confidence_threshold = sentiment_config.HIGH_CONFIDENCE_THRESHOLD
        
        # Performance settings
        self.max_articles = sentiment_config.MAX_ARTICLES_PER_SYMBOL
        self.min_articles = sentiment_config.MIN_ARTICLES_FOR_ANALYSIS
        self.timeout_seconds = sentiment_config.TIMEOUT_SECONDS
        
        # Symbol mappings for currency pairs
        self.currency_mappings = {
            'EURUSD': ['EUR', 'USD', 'ECB', 'FED', 'EURO', 'DOLLAR'],
            'GBPUSD': ['GBP', 'USD', 'BOE', 'FED', 'POUND', 'STERLING', 'DOLLAR'],
            'USDJPY': ['USD', 'JPY', 'FED', 'BOJ', 'DOLLAR', 'YEN'],
            'USDCHF': ['USD', 'CHF', 'FED', 'SNB', 'DOLLAR', 'FRANC'],
            'AUDUSD': ['AUD', 'USD', 'RBA', 'FED', 'DOLLAR', 'AUSSIE'],
            'USDCAD': ['USD', 'CAD', 'FED', 'BOC', 'DOLLAR', 'LOONIE'],
            'NZDUSD': ['NZD', 'USD', 'RBNZ', 'FED', 'DOLLAR', 'KIWI'],
            'EURJPY': ['EUR', 'JPY', 'ECB', 'BOJ', 'EURO', 'YEN'],
            'GBPJPY': ['GBP', 'JPY', 'BOE', 'BOJ', 'POUND', 'YEN'],
            'EURGBP': ['EUR', 'GBP', 'ECB', 'BOE', 'EURO', 'POUND'],
            'AUDJPY': ['AUD', 'JPY', 'RBA', 'BOJ', 'AUSSIE', 'YEN'],
            'EURAUD': ['EUR', 'AUD', 'ECB', 'RBA', 'EURO', 'AUSSIE'],
            'BTCUSD': ['BTC', 'BITCOIN', 'CRYPTO', 'CRYPTOCURRENCY'],
            # Add more pairs as needed
        }
        
        self.logger.info(f"Sentiment factor service initialized - Enabled: {self.sentiment_enabled}, "
                        f"Weight: {self.sentiment_weight}, Lookback: {self.lookback_hours}h")
    
    async def get_sentiment_factor(self, symbol: str, db: Session) -> Dict[str, Any]:
        """
        Calculate sentiment factor for a given trading symbol
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD', 'BTCUSD')
            db: Database session
            
        Returns:
            Dict containing sentiment score, impact, confidence, and reasoning
        """
        if not self.sentiment_enabled:
            return self._neutral_result("Sentiment analysis disabled")
        
        try:
            # Get recent news articles relevant to the symbol
            relevant_articles = await self._get_relevant_news(symbol, db)
            
            if not relevant_articles:
                return self._neutral_result(f"No recent news found for {symbol}")
            
            # Calculate aggregated sentiment
            sentiment_data = await self._calculate_aggregated_sentiment(
                symbol, relevant_articles, db
            )
            
            # Apply time decay and confidence weighting
            final_sentiment = self._apply_weighting_factors(sentiment_data)
            
            # Calculate impact on signal confidence
            impact_factor = self._calculate_confidence_impact(final_sentiment['score'])
            
            result = {
                'sentiment_score': round(final_sentiment['score'], 3),
                'sentiment_confidence': round(final_sentiment['confidence'], 3),
                'sentiment_impact': round(impact_factor, 3),
                'sentiment_label': self._get_sentiment_label(final_sentiment['score']),
                'articles_analyzed': len(relevant_articles),
                'reasoning': self._generate_reasoning(final_sentiment, impact_factor, len(relevant_articles)),
                'timestamp': datetime.utcnow().isoformat(),
                'enabled': True,
                'symbol': symbol
            }
            
            self.logger.debug(f"Sentiment factor for {symbol}: {result['sentiment_score']:.3f} "
                            f"(impact: {result['sentiment_impact']:.3f})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating sentiment factor for {symbol}: {e}")
            return self._neutral_result(f"Error: {str(e)}")
    
    async def _get_relevant_news(self, symbol: str, db: Session) -> List[NewsArticle]:
        """Get news articles relevant to the trading symbol"""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.lookback_hours)
        
        # Get symbol-specific keywords
        keywords = self.currency_mappings.get(symbol, [symbol])
        
        try:
            # Query for articles that:
            # 1. Are recent (within lookback period)
            # 2. Are relevant to trading
            # 3. Contain symbol-specific keywords or have the symbol in their symbols JSON field
            query = db.query(NewsArticle).filter(
                and_(
                    NewsArticle.published_at >= cutoff_time,
                    NewsArticle.is_relevant == True,
                    or_(
                        # Check if symbol is in the symbols JSON array
                        NewsArticle.symbols.contains([symbol]),
                        # Check if any keyword appears in title or content
                        or_(*[
                            or_(
                                NewsArticle.title.ilike(f'%{keyword}%'),
                                NewsArticle.content.ilike(f'%{keyword}%')
                            ) for keyword in keywords
                        ]) if keywords else NewsArticle.id == None  # Fallback condition
                    )
                )
            ).order_by(desc(NewsArticle.published_at))
            
            articles = query.limit(self.max_articles).all()  # Limit to recent articles
            
            self.logger.debug(f"Found {len(articles)} relevant articles for {symbol} "
                            f"in last {self.lookback_hours} hours")
            
            return articles
            
        except Exception as e:
            self.logger.error(f"Error querying relevant news for {symbol}: {e}")
            return []
    
    async def _calculate_aggregated_sentiment(
        self, 
        symbol: str, 
        articles: List[NewsArticle], 
        db: Session
    ) -> Dict[str, float]:
        """Calculate aggregated sentiment from news articles"""
        
        if not articles:
            return {'score': 0.0, 'confidence': 0.0, 'count': 0}
        
        weighted_scores = []
        confidence_weights = []
        
        for article in articles:
            # Get sentiment analysis for this article (prefer combined method)
            sentiment = db.query(NewsSentiment).filter(
                and_(
                    NewsSentiment.news_article_id == article.id,
                    NewsSentiment.analyzer_type == 'combined'
                )
            ).first()
            
            # Fallback to other analyzers if combined not available
            if not sentiment:
                sentiment = db.query(NewsSentiment).filter(
                    NewsSentiment.news_article_id == article.id
                ).order_by(desc(NewsSentiment.confidence_score)).first()
            
            if sentiment:
                # Calculate time decay factor (more recent = higher weight)
                hours_old = (datetime.utcnow() - article.published_at).total_seconds() / 3600
                time_decay = max(0.1, (1.0 - self.recency_decay) ** hours_old)
                
                # Weight by confidence and recency
                final_weight = sentiment.confidence_score * time_decay
                
                weighted_scores.append(sentiment.sentiment_score * final_weight)
                confidence_weights.append(final_weight)
                
                self.logger.debug(f"Article sentiment: {sentiment.sentiment_score:.3f}, "
                                f"confidence: {sentiment.confidence_score:.3f}, "
                                f"age: {hours_old:.1f}h, weight: {final_weight:.3f}")
        
        if not weighted_scores:
            return {'score': 0.0, 'confidence': 0.0, 'count': 0}
        
        # Calculate weighted average
        total_weight = sum(confidence_weights)
        if total_weight > 0:
            aggregated_score = sum(weighted_scores) / total_weight
            avg_confidence = total_weight / len(confidence_weights)  # Normalized confidence
        else:
            aggregated_score = 0.0
            avg_confidence = 0.0
        
        return {
            'score': aggregated_score,
            'confidence': min(avg_confidence, 1.0),  # Cap at 1.0
            'count': len(weighted_scores)
        }
    
    def _apply_weighting_factors(self, sentiment_data: Dict[str, float]) -> Dict[str, float]:
        """Apply additional weighting factors based on confidence and sample size"""
        
        score = sentiment_data['score']
        confidence = sentiment_data['confidence']
        count = sentiment_data['count']
        
        # Sample size factor (more articles = higher confidence)
        sample_factor = min(1.0, count / 10.0)  # Plateau at 10 articles
        
        # Confidence adjustment
        adjusted_confidence = confidence * sample_factor
        
        # Dampen extreme scores if confidence is low
        if adjusted_confidence < 0.5:
            score *= adjusted_confidence * 2  # Scale down if low confidence
        
        return {
            'score': score,
            'confidence': adjusted_confidence
        }
    
    def _calculate_confidence_impact(self, sentiment_score: float) -> float:
        """Calculate the impact on signal confidence based on sentiment score"""
        
        # Apply sentiment weight to limit maximum impact
        max_impact = self.sentiment_weight
        
        # Strong positive sentiment boosts confidence
        if sentiment_score > self.positive_threshold:
            if sentiment_score > self.high_confidence_threshold:
                return max_impact  # Full positive impact
            else:
                # Linear scaling between threshold and high confidence
                factor = (sentiment_score - self.positive_threshold) / (self.high_confidence_threshold - self.positive_threshold)
                return factor * max_impact
        
        # Strong negative sentiment reduces confidence
        elif sentiment_score < self.negative_threshold:
            if sentiment_score < -self.high_confidence_threshold:
                return -max_impact  # Full negative impact
            else:
                # Linear scaling between threshold and high confidence
                factor = (abs(sentiment_score) - self.positive_threshold) / (self.high_confidence_threshold - self.positive_threshold)
                return -factor * max_impact
        
        # Neutral sentiment has minimal impact
        else:
            return 0.0
    
    def _get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to readable label"""
        if score > self.positive_threshold:
            return "POSITIVE"
        elif score < self.negative_threshold:
            return "NEGATIVE"
        else:
            return "NEUTRAL"
    
    def _generate_reasoning(self, sentiment_data: Dict, impact: float, article_count: int) -> str:
        """Generate human-readable reasoning for sentiment impact"""
        
        score = sentiment_data['score']
        confidence = sentiment_data['confidence']
        label = self._get_sentiment_label(score)
        
        if article_count == 0:
            return "No recent news available for sentiment analysis"
        
        impact_desc = "neutral"
        if abs(impact) > 0.05:  # 5% impact threshold
            impact_desc = "positive boost" if impact > 0 else "negative reduction"
        
        confidence_desc = "high" if confidence > 0.7 else "moderate" if confidence > 0.4 else "low"
        
        return (f"{label.lower()} sentiment (score: {score:.2f}) from {article_count} "
                f"recent articles with {confidence_desc} confidence, "
                f"providing {impact_desc} to signal confidence ({impact:+.1%})")
    
    def _neutral_result(self, reason: str) -> Dict[str, Any]:
        """Return neutral sentiment result"""
        return {
            'sentiment_score': 0.0,
            'sentiment_confidence': 0.0,
            'sentiment_impact': 0.0,
            'sentiment_label': 'NEUTRAL',
            'articles_analyzed': 0,
            'reasoning': reason,
            'timestamp': datetime.utcnow().isoformat(),
            'enabled': self.sentiment_enabled,
            'symbol': None
        }
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get current sentiment factor configuration"""
        return {
            'enabled': self.sentiment_enabled,
            'weight': self.sentiment_weight,
            'lookback_hours': self.lookback_hours,
            'positive_threshold': self.positive_threshold,
            'negative_threshold': self.negative_threshold,
            'high_confidence_threshold': self.high_confidence_threshold,
            'recency_decay': self.recency_decay
        }
    
    def update_configuration(self, **kwargs) -> Dict[str, Any]:
        """Update sentiment factor configuration"""
        updated = {}
        
        if 'enabled' in kwargs:
            self.sentiment_enabled = bool(kwargs['enabled'])
            updated['enabled'] = self.sentiment_enabled
        
        if 'weight' in kwargs:
            self.sentiment_weight = max(0.0, min(1.0, float(kwargs['weight'])))  # Clamp 0-1
            updated['weight'] = self.sentiment_weight
        
        if 'lookback_hours' in kwargs:
            self.lookback_hours = max(1, int(kwargs['lookback_hours']))  # At least 1 hour
            updated['lookback_hours'] = self.lookback_hours
        
        self.logger.info(f"Sentiment configuration updated: {updated}")
        return updated


# Global instance
sentiment_factor_service = SentimentFactorService()