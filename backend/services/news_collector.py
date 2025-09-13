"""
Comprehensive News Collection Service

This service orchestrates news collection from multiple providers,
handles sentiment analysis, and manages the news database.
"""

import asyncio
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc

from ..models import NewsArticle, NewsSentiment
from ..database import SessionLocal
from ..logs.logger import get_logger
from ..providers.alphavantage import AlphaVantageProvider
from ..providers.finnhub_provider import FinnhubProvider
from .sentiment_analyzer import SentimentAnalyzer

logger = get_logger(__name__)

class NewsCollector:
    """Comprehensive news collection and management service"""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        
        # Initialize data providers
        self.alphavantage = AlphaVantageProvider()
        self.finnhub = FinnhubProvider()
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # News collection settings
        self.forex_symbols = [
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD',
            'NZDUSD', 'EURJPY', 'GBPJPY', 'EURGBP', 'AUDJPY', 'EURAUD'
        ]
        
        self.categories = ['general', 'forex', 'crypto', 'economy', 'finance']
        
        # Deduplication settings
        self.dedup_threshold_hours = 24  # Consider articles within 24h as potential duplicates
        
        self.logger.info("News collector initialized with AlphaVantage, Finnhub, and sentiment analysis")
    
    async def collect_all_news(
        self, 
        force_refresh: bool = False,
        symbols: Optional[List[str]] = None,
        categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Collect news from all available providers
        
        Args:
            force_refresh: Skip recent article checks
            symbols: Specific symbols to collect news for
            categories: Specific categories to collect
            
        Returns:
            Collection summary with counts and any errors
        """
        start_time = datetime.utcnow()
        
        # Use provided filters or defaults
        target_symbols = symbols or self.forex_symbols
        target_categories = categories or self.categories
        
        summary = {
            'start_time': start_time.isoformat(),
            'providers_used': [],
            'total_collected': 0,
            'total_stored': 0,
            'total_analyzed': 0,
            'errors': [],
            'by_category': {},
            'by_provider': {},
            'processing_time_seconds': 0
        }
        
        try:
            # Collect news from all available providers
            collection_tasks = []
            
            # AlphaVantage news collection
            if self.alphavantage.enabled:
                summary['providers_used'].append('alphavantage')
                summary['by_provider']['alphavantage'] = {'collected': 0, 'stored': 0, 'errors': 0}
                
                for category in target_categories:
                    collection_tasks.append(
                        self._collect_from_alphavantage(category, force_refresh, summary)
                    )
                
                # Symbol-specific news
                for symbol in target_symbols[:5]:  # Limit to avoid rate limits
                    collection_tasks.append(
                        self._collect_symbol_news_alphavantage(symbol, force_refresh, summary)
                    )
            
            # Finnhub news collection
            if self.finnhub.is_available():
                summary['providers_used'].append('finnhub')
                summary['by_provider']['finnhub'] = {'collected': 0, 'stored': 0, 'errors': 0}
                
                for category in target_categories:
                    collection_tasks.append(
                        self._collect_from_finnhub(category, force_refresh, summary)
                    )
                
                # Market news with sentiment
                collection_tasks.append(
                    self._collect_market_news_finnhub(force_refresh, summary)
                )
            
            # Execute all collection tasks concurrently
            if collection_tasks:
                await asyncio.gather(*collection_tasks, return_exceptions=True)
            else:
                self.logger.warning("No news providers available for collection")
                summary['errors'].append("No news providers available")
            
            # Calculate processing time
            end_time = datetime.utcnow()
            summary['processing_time_seconds'] = (end_time - start_time).total_seconds()
            summary['end_time'] = end_time.isoformat()
            
            self.logger.info(
                f"News collection completed: {summary['total_stored']} articles stored, "
                f"{summary['total_analyzed']} analyzed in {summary['processing_time_seconds']:.2f}s"
            )
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error in news collection: {e}")
            summary['errors'].append(f"Collection error: {str(e)}")
            return summary
    
    async def _collect_from_alphavantage(
        self, 
        category: str, 
        force_refresh: bool,
        summary: Dict[str, Any]
    ) -> None:
        """Collect news from AlphaVantage by category"""
        try:
            articles = await self.alphavantage.get_news(category, limit=50)
            if not articles:
                return
            
            summary['by_provider']['alphavantage']['collected'] += len(articles)
            summary['total_collected'] += len(articles)
            
            # Initialize category tracking
            if category not in summary['by_category']:
                summary['by_category'][category] = {'collected': 0, 'stored': 0}
            
            summary['by_category'][category]['collected'] += len(articles)
            
            # Store and analyze articles
            for article_data in articles:
                try:
                    stored = await self._store_article(article_data, 'alphavantage', force_refresh)
                    if stored:
                        summary['by_provider']['alphavantage']['stored'] += 1
                        summary['by_category'][category]['stored'] += 1
                        summary['total_stored'] += 1
                        
                        # Run sentiment analysis
                        analyzed = await self._analyze_article_sentiment(stored)
                        if analyzed:
                            summary['total_analyzed'] += 1
                            
                except Exception as e:
                    self.logger.error(f"Error storing AlphaVantage article: {e}")
                    summary['by_provider']['alphavantage']['errors'] += 1
                    
        except Exception as e:
            self.logger.error(f"Error collecting from AlphaVantage {category}: {e}")
            summary['errors'].append(f"AlphaVantage {category}: {str(e)}")
    
    async def _collect_symbol_news_alphavantage(
        self, 
        symbol: str, 
        force_refresh: bool,
        summary: Dict[str, Any]
    ) -> None:
        """Collect symbol-specific news from AlphaVantage"""
        try:
            articles = await self.alphavantage.get_symbol_news(symbol, limit=20)
            if not articles:
                return
            
            summary['by_provider']['alphavantage']['collected'] += len(articles)
            summary['total_collected'] += len(articles)
            
            # Store and analyze articles
            for article_data in articles:
                try:
                    # Add symbol to article data
                    if 'symbols' not in article_data:
                        article_data['symbols'] = []
                    if symbol not in article_data['symbols']:
                        article_data['symbols'].append(symbol)
                    
                    stored = await self._store_article(article_data, 'alphavantage', force_refresh)
                    if stored:
                        summary['by_provider']['alphavantage']['stored'] += 1
                        summary['total_stored'] += 1
                        
                        # Run sentiment analysis
                        analyzed = await self._analyze_article_sentiment(stored)
                        if analyzed:
                            summary['total_analyzed'] += 1
                            
                except Exception as e:
                    self.logger.error(f"Error storing AlphaVantage symbol article: {e}")
                    summary['by_provider']['alphavantage']['errors'] += 1
                    
        except Exception as e:
            self.logger.error(f"Error collecting symbol news from AlphaVantage {symbol}: {e}")
            summary['errors'].append(f"AlphaVantage {symbol}: {str(e)}")
    
    async def _collect_from_finnhub(
        self, 
        category: str, 
        force_refresh: bool,
        summary: Dict[str, Any]
    ) -> None:
        """Collect news from Finnhub by category"""
        try:
            articles = await self.finnhub.get_news(category, limit=50)
            if not articles:
                return
            
            summary['by_provider']['finnhub']['collected'] += len(articles)
            summary['total_collected'] += len(articles)
            
            # Initialize category tracking
            if category not in summary['by_category']:
                summary['by_category'][category] = {'collected': 0, 'stored': 0}
            
            summary['by_category'][category]['collected'] += len(articles)
            
            # Store and analyze articles
            for article_data in articles:
                try:
                    stored = await self._store_article(article_data, 'finnhub', force_refresh)
                    if stored:
                        summary['by_provider']['finnhub']['stored'] += 1
                        summary['by_category'][category]['stored'] += 1
                        summary['total_stored'] += 1
                        
                        # Run sentiment analysis
                        analyzed = await self._analyze_article_sentiment(stored)
                        if analyzed:
                            summary['total_analyzed'] += 1
                            
                except Exception as e:
                    self.logger.error(f"Error storing Finnhub article: {e}")
                    summary['by_provider']['finnhub']['errors'] += 1
                    
        except Exception as e:
            self.logger.error(f"Error collecting from Finnhub {category}: {e}")
            summary['errors'].append(f"Finnhub {category}: {str(e)}")
    
    async def _collect_market_news_finnhub(
        self, 
        force_refresh: bool,
        summary: Dict[str, Any]
    ) -> None:
        """Collect market news with sentiment from Finnhub"""
        try:
            articles = await self.finnhub.get_market_news_with_sentiment(limit=30)
            if not articles:
                return
            
            summary['by_provider']['finnhub']['collected'] += len(articles)
            summary['total_collected'] += len(articles)
            
            # Store and analyze articles
            for article_data in articles:
                try:
                    stored = await self._store_article(article_data, 'finnhub', force_refresh)
                    if stored:
                        summary['by_provider']['finnhub']['stored'] += 1
                        summary['total_stored'] += 1
                        
                        # Run sentiment analysis (may already have basic sentiment)
                        analyzed = await self._analyze_article_sentiment(stored)
                        if analyzed:
                            summary['total_analyzed'] += 1
                            
                except Exception as e:
                    self.logger.error(f"Error storing Finnhub market article: {e}")
                    summary['by_provider']['finnhub']['errors'] += 1
                    
        except Exception as e:
            self.logger.error(f"Error collecting market news from Finnhub: {e}")
            summary['errors'].append(f"Finnhub market news: {str(e)}")
    
    async def _store_article(
        self, 
        article_data: Dict[str, Any], 
        provider: str,
        force_refresh: bool = False
    ) -> Optional[NewsArticle]:
        """
        Store news article in database with deduplication
        
        Args:
            article_data: Article data from provider
            provider: Provider name
            force_refresh: Skip duplicate checks
            
        Returns:
            Stored NewsArticle or None if duplicate/error
        """
        try:
            db = SessionLocal()
            
            # Extract and normalize article data
            url = article_data.get('url', '').strip()
            title = article_data.get('title', '').strip()
            
            if not url or not title:
                self.logger.warning(f"Skipping article with missing URL or title")
                return None
            
            # Check for existing article by URL (primary deduplication)
            if not force_refresh:
                existing = db.query(NewsArticle).filter(NewsArticle.url == url).first()
                if existing:
                    self.logger.debug(f"Article already exists: {url}")
                    db.close()
                    return existing
            
            # Parse published date
            published_at = self._parse_published_date(article_data)
            if not published_at:
                published_at = datetime.utcnow()
            
            # Extract symbols list
            symbols = article_data.get('symbols', [])
            if isinstance(symbols, str):
                symbols = [symbols]
            elif not isinstance(symbols, list):
                symbols = []
            
            # Create new article
            article = NewsArticle(
                title=title[:500],  # Truncate to fit database field
                summary=article_data.get('summary', '')[:2000] if article_data.get('summary') else None,
                content=article_data.get('content', '')[:10000] if article_data.get('content') else None,
                url=url,
                source=provider,
                published_at=published_at,
                category=article_data.get('category', 'general'),
                symbols=symbols,
                is_relevant=True  # Provider-filtered articles are considered relevant
            )
            
            db.add(article)
            db.commit()
            db.refresh(article)
            
            self.logger.debug(f"Stored article: {article.title[:50]}...")
            db.close()
            return article
            
        except Exception as e:
            self.logger.error(f"Error storing article: {e}")
            if 'db' in locals():
                db.rollback()
                db.close()
            return None
    
    def _parse_published_date(self, article_data: Dict[str, Any]) -> Optional[datetime]:
        """Parse published date from various provider formats"""
        try:
            # Try different date field names
            for field in ['published_at', 'datetime', 'timestamp', 'time_published']:
                if field in article_data:
                    value = article_data[field]
                    
                    if isinstance(value, str):
                        # Try ISO format
                        try:
                            return datetime.fromisoformat(value.replace('Z', '+00:00'))
                        except:
                            continue
                    elif isinstance(value, (int, float)):
                        # Unix timestamp
                        return datetime.fromtimestamp(value)
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error parsing published date: {e}")
            return None
    
    async def _analyze_article_sentiment(self, article: NewsArticle) -> bool:
        """
        Run comprehensive sentiment analysis on an article
        
        Args:
            article: NewsArticle instance
            
        Returns:
            True if analysis was successful
        """
        try:
            db = SessionLocal()
            
            # Check if sentiment analysis already exists
            existing_sentiment = db.query(NewsSentiment).filter(
                and_(
                    NewsSentiment.news_article_id == article.id,
                    NewsSentiment.analyzer_type == 'combined'
                )
            ).first()
            
            if existing_sentiment:
                self.logger.debug(f"Sentiment analysis already exists for article {article.id}")
                db.close()
                return True
            
            # Prepare text for analysis
            text_to_analyze = f"{article.title or ''} {article.summary or ''}"
            if not text_to_analyze.strip():
                self.logger.warning(f"No text content for sentiment analysis: article {article.id}")
                db.close()
                return False
            
            # Run sentiment analysis
            sentiment_result = self.sentiment_analyzer.analyze_sentiment(
                text_to_analyze, 
                method="combined"
            )
            
            if not sentiment_result or 'score' not in sentiment_result:
                self.logger.error(f"Sentiment analysis failed for article {article.id}")
                db.close()
                return False
            
            # Store sentiment result
            sentiment = NewsSentiment(
                news_article_id=article.id,
                analyzer_type='combined',
                sentiment_score=sentiment_result['score'],
                confidence_score=sentiment_result.get('confidence', 0.5),
                sentiment_label=sentiment_result.get('label', 'NEUTRAL').upper()
            )
            
            db.add(sentiment)
            db.commit()
            
            self.logger.debug(f"Stored sentiment analysis for article {article.id}: {sentiment.sentiment_label}")
            db.close()
            return True
            
        except Exception as e:
            self.logger.error(f"Error analyzing article sentiment: {e}")
            if 'db' in locals():
                db.rollback()
                db.close()
            return False
    
    def get_news_articles(
        self, 
        db: Session,
        symbol: Optional[str] = None,
        category: Optional[str] = None,
        days: int = 7,
        limit: int = 50,
        include_sentiment: bool = True
    ) -> List[NewsArticle]:
        """
        Retrieve news articles with optional filtering
        
        Args:
            db: Database session
            symbol: Filter by trading symbol
            category: Filter by news category
            days: Look back days
            limit: Maximum articles to return
            include_sentiment: Include sentiment data
            
        Returns:
            List of NewsArticle objects
        """
        try:
            # Build base query
            query = db.query(NewsArticle).filter(NewsArticle.is_relevant == True)
            
            # Date filter
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            query = query.filter(NewsArticle.published_at >= cutoff_date)
            
            # Symbol filter (check JSON array)
            if symbol:
                query = query.filter(
                    or_(
                        NewsArticle.symbols.contains([symbol.upper()]),
                        NewsArticle.title.ilike(f'%{symbol}%')
                    )
                )
            
            # Category filter
            if category:
                query = query.filter(NewsArticle.category == category.lower())
            
            # Include sentiment data if requested
            if include_sentiment:
                from sqlalchemy.orm import joinedload
                query = query.options(joinedload(NewsArticle.sentiments))
            
            # Order by published date and limit
            articles = query.order_by(desc(NewsArticle.published_at)).limit(limit).all()
            
            self.logger.debug(f"Retrieved {len(articles)} articles with filters: symbol={symbol}, category={category}, days={days}")
            return articles
            
        except Exception as e:
            self.logger.error(f"Error retrieving news articles: {e}")
            return []
    
    def get_sentiment_summary(
        self, 
        db: Session,
        symbol: Optional[str] = None,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Get overall sentiment summary for a timeframe
        
        Args:
            db: Database session
            symbol: Optional symbol filter
            days: Look back days
            
        Returns:
            Sentiment summary dictionary
        """
        try:
            # Get articles in timeframe
            articles = self.get_news_articles(
                db, symbol=symbol, days=days, limit=1000, include_sentiment=True
            )
            
            if not articles:
                return {
                    'overall_sentiment': 'NEUTRAL',
                    'overall_score': 0.0,
                    'confidence': 0.0,
                    'total_articles': 0,
                    'positive_articles': 0,
                    'negative_articles': 0,
                    'neutral_articles': 0,
                    'timeframe': f"{days}d",
                    'by_symbol': {},
                    'by_source': {}
                }
            
            # Aggregate sentiment data
            total_articles = len(articles)
            sentiment_scores = []
            positive_count = 0
            negative_count = 0
            neutral_count = 0
            
            by_symbol = {}
            by_source = {}
            
            for article in articles:
                # Get the latest/best sentiment for this article
                article_sentiment = None
                if article.sentiments:
                    # Prefer combined analysis
                    for sent in article.sentiments:
                        if sent.analyzer_type == 'combined':
                            article_sentiment = sent
                            break
                    # Fallback to any sentiment
                    if not article_sentiment:
                        article_sentiment = article.sentiments[0]
                
                if article_sentiment:
                    sentiment_scores.append(article_sentiment.sentiment_score)
                    
                    # Count by label
                    if article_sentiment.sentiment_label == 'POSITIVE':
                        positive_count += 1
                    elif article_sentiment.sentiment_label == 'NEGATIVE':
                        negative_count += 1
                    else:
                        neutral_count += 1
                    
                    # Aggregate by symbol
                    if article.symbols:
                        for sym in article.symbols:
                            if sym not in by_symbol:
                                by_symbol[sym] = {'scores': [], 'count': 0}
                            by_symbol[sym]['scores'].append(article_sentiment.sentiment_score)
                            by_symbol[sym]['count'] += 1
                    
                    # Aggregate by source
                    source = article.source
                    if source not in by_source:
                        by_source[source] = {'scores': [], 'count': 0}
                    by_source[source]['scores'].append(article_sentiment.sentiment_score)
                    by_source[source]['count'] += 1
            
            # Calculate overall metrics
            if sentiment_scores:
                overall_score = sum(sentiment_scores) / len(sentiment_scores)
                confidence = min(len(sentiment_scores) / 20.0, 1.0)  # More articles = higher confidence
                
                # Determine overall sentiment
                if overall_score > 0.1:
                    overall_sentiment = 'POSITIVE'
                elif overall_score < -0.1:
                    overall_sentiment = 'NEGATIVE'
                else:
                    overall_sentiment = 'NEUTRAL'
            else:
                overall_score = 0.0
                confidence = 0.0
                overall_sentiment = 'NEUTRAL'
            
            # Process by_symbol aggregates
            for sym in by_symbol:
                scores = by_symbol[sym]['scores']
                by_symbol[sym] = {
                    'average_score': sum(scores) / len(scores) if scores else 0,
                    'total_articles': len(scores),
                    'sentiment': 'POSITIVE' if sum(scores) / len(scores) > 0.1 else 'NEGATIVE' if sum(scores) / len(scores) < -0.1 else 'NEUTRAL'
                }
            
            # Process by_source aggregates
            for source in by_source:
                scores = by_source[source]['scores']
                by_source[source] = {
                    'average_score': sum(scores) / len(scores) if scores else 0,
                    'total_articles': len(scores),
                    'sentiment': 'POSITIVE' if sum(scores) / len(scores) > 0.1 else 'NEGATIVE' if sum(scores) / len(scores) < -0.1 else 'NEUTRAL'
                }
            
            return {
                'overall_sentiment': overall_sentiment,
                'overall_score': round(overall_score, 4),
                'confidence': round(confidence, 4),
                'total_articles': total_articles,
                'positive_articles': positive_count,
                'negative_articles': negative_count,
                'neutral_articles': neutral_count,
                'timeframe': f"{days}d",
                'by_symbol': by_symbol,
                'by_source': by_source
            }
            
        except Exception as e:
            self.logger.error(f"Error generating sentiment summary: {e}")
            return {
                'overall_sentiment': 'NEUTRAL',
                'overall_score': 0.0,
                'confidence': 0.0,
                'total_articles': 0,
                'positive_articles': 0,
                'negative_articles': 0,
                'neutral_articles': 0,
                'timeframe': f"{days}d",
                'error': str(e)
            }

# Singleton instance
news_collector = NewsCollector()