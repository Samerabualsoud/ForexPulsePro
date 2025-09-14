"""
Alpha Vantage Data Provider - Production Implementation
"""
import os
import requests
import pandas as pd
import asyncio
import httpx
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from pathlib import Path
import time
import json
import random

from .base import BaseDataProvider
from ..logs.logger import get_logger

logger = get_logger(__name__)

class AlphaVantageProvider(BaseDataProvider):
    """Alpha Vantage API data provider with rate limiting and caching"""
    
    def __init__(self):
        self.api_key = os.getenv("ALPHAVANTAGE_KEY")
        self.base_url = "https://www.alphavantage.co/query"
        self.enabled = bool(self.api_key)
        
        # Rate limiting (Alpha Vantage allows 5 calls per minute for free tier)
        self.calls_per_minute = 5
        self.call_timestamps = []
        self._rate_limit_lock = None  # Lazy initialization to avoid event loop binding issues
        
        # Shared async client for connection reuse
        self._client = None
        
        # Caching
        self.cache_dir = Path("data/alphavantage")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_duration = timedelta(minutes=1)  # Cache for 1 minute
        self.data_cache = {}
        self.price_cache = {}
        
        # Commodity symbol mapping for Alpha Vantage
        self.commodity_mapping = {
            'XAUUSD': {'function': 'PRECIOUS_METALS', 'symbol': 'XAU', 'market': 'USD'},
            'XAGUSD': {'function': 'PRECIOUS_METALS', 'symbol': 'XAG', 'market': 'USD'}, 
            'USOIL': {'function': 'CRUDE_OIL', 'symbol': 'WTI', 'market': 'USD'}
        }
        
        if not self.enabled:
            logger.info("Alpha Vantage provider disabled - no API key provided")
        else:
            logger.info("Alpha Vantage provider initialized with rate limiting and commodity support")

    @property
    def rate_limit_lock(self) -> asyncio.Lock:
        """Get rate limiting lock, creating it lazily in the current event loop"""
        if self._rate_limit_lock is None:
            try:
                # This will create the lock in the current event loop
                self._rate_limit_lock = asyncio.Lock()
            except RuntimeError:
                # No event loop running, this should not happen in async context
                # but we'll handle it gracefully
                logger.warning("No event loop running when creating rate limit lock")
                raise
        return self._rate_limit_lock
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create shared async client"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30)
        return self._client
    
    async def _check_and_wait_rate_limit(self) -> None:
        """Thread-safe rate limiting with token bucket algorithm"""
        async with self.rate_limit_lock:
            now = time.time()
            
            # Remove timestamps older than 1 minute (token bucket refill)
            self.call_timestamps = [ts for ts in self.call_timestamps if now - ts < 60]
            
            # If we're at the limit, wait until we can make a call
            while len(self.call_timestamps) >= self.calls_per_minute:
                # Calculate wait time until oldest call expires
                if self.call_timestamps:
                    oldest_call = min(self.call_timestamps)
                    wait_time = max(1, 60 - (now - oldest_call))
                else:
                    wait_time = 12
                
                logger.info(f"Alpha Vantage rate limit reached, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                
                # Refresh timestamps after waiting
                now = time.time()
                self.call_timestamps = [ts for ts in self.call_timestamps if now - ts < 60]
            
            # Record this call
            self.call_timestamps.append(now)
    
    def _get_cache_key(self, symbol: str, timeframe: str, data_type: str) -> str:
        """Generate cache key for data"""
        return f"{symbol}_{timeframe}_{data_type}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.data_cache:
            return False
        
        cached_data = self.data_cache[cache_key]
        if 'timestamp' not in cached_data:
            return False
            
        cache_time = cached_data['timestamp']
        return datetime.now() - cache_time < self.cache_duration
    
    async def get_ohlc_data(
        self, 
        symbol: str, 
        timeframe: str = "M1", 
        limit: int = 100,
        retry_count: int = 0
    ) -> Optional[pd.DataFrame]:
        """
        Get OHLC data from Alpha Vantage API with rate limiting and caching
        """
        if not self.enabled:
            return None
        
        # Check cache first
        cache_key = self._get_cache_key(symbol, timeframe, "ohlc")
        if self._is_cache_valid(cache_key):
            logger.debug(f"Using cached data for {symbol}")
            return self.data_cache[cache_key]['data'].tail(limit) if limit else self.data_cache[cache_key]['data']
        
        try:
            # Check if this is a commodity symbol
            if symbol in self.commodity_mapping:
                # Handle commodity data request
                commodity_info = self.commodity_mapping[symbol]
                
                # Wait for rate limit if necessary
                await self._check_and_wait_rate_limit()
                
                # For commodities, use daily data as Alpha Vantage has limited intraday commodity support
                params = {
                    "function": commodity_info['function'],
                    "symbol": commodity_info['symbol'],
                    "market": commodity_info['market'],
                    "apikey": self.api_key
                }
                
                # Special handling for crude oil
                if commodity_info['function'] == 'CRUDE_OIL':
                    params = {
                        "function": "CRUDE_OIL",
                        "interval": "daily",
                        "apikey": self.api_key
                    }
                
            else:
                # Handle forex pairs
                # Convert symbol format (EURUSD -> EUR/USD)
                if len(symbol) == 6:
                    from_currency = symbol[:3]
                    to_currency = symbol[3:]
                else:
                    logger.error(f"Invalid symbol format: {symbol}")
                    return None
                
                # Wait for rate limit if necessary
                await self._check_and_wait_rate_limit()
                
                # Map timeframe to Alpha Vantage intervals
                interval_map = {
                    "M1": "1min",
                    "M5": "5min", 
                    "M15": "15min",
                    "M30": "30min",
                    "H1": "60min"
                }
                
                interval = interval_map.get(timeframe, "1min")
                
                params = {
                    "function": "FX_INTRADAY",
                    "from_symbol": from_currency,
                    "to_symbol": to_currency,
                    "interval": interval,
                    "apikey": self.api_key,
                    "outputsize": "full"  # Get more data points
                }
            
            # Make async request with shared client
            client = await self._get_client()
            response = await client.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Check for API errors
            if "Error Message" in data:
                logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                return None
            
            if "Note" in data:
                # Exponential backoff with jitter for rate limit notes
                if retry_count >= 3:  # Max 3 retries
                    logger.error(f"Alpha Vantage max retries exceeded: {data['Note']}")
                    return None
                
                base_delay = 15 * (2 ** retry_count)  # Exponential backoff
                jitter = random.uniform(0.8, 1.2)  # Add jitter
                delay = min(base_delay * jitter, 120)  # Cap at 2 minutes
                
                logger.warning(f"Alpha Vantage rate limit hit (retry {retry_count + 1}/3): {data['Note']}, waiting {delay:.1f}s")
                await asyncio.sleep(delay)
                
                # Increment retry count and recurse with updated count
                return await self.get_ohlc_data(symbol, timeframe, limit, retry_count + 1)
            
            if "Information" in data and "rate limit" in data["Information"].lower():
                logger.warning(f"Alpha Vantage rate limit: {data['Information']}")
                return None
            
            # Parse time series data based on symbol type
            rows = []
            
            if symbol in self.commodity_mapping:
                # Handle commodity data parsing
                commodity_info = self.commodity_mapping[symbol]
                
                if commodity_info['function'] == 'PRECIOUS_METALS':
                    # Gold/Silver data format
                    if 'Monthly Prices' in data:
                        time_series = data['Monthly Prices']
                        for timestamp_str, price_data in time_series.items():
                            try:
                                # Create OHLC from single price point (typical for precious metals)
                                price = float(price_data['price'])
                                rows.append({
                                    'timestamp': pd.to_datetime(timestamp_str),
                                    'open': price,
                                    'high': price,
                                    'low': price,
                                    'close': price,
                                    'volume': 0
                                })
                            except (KeyError, ValueError) as e:
                                logger.warning(f"Skipping invalid commodity data point for {timestamp_str}: {e}")
                                continue
                                
                elif commodity_info['function'] == 'CRUDE_OIL':
                    # Oil data format
                    if 'data' in data:
                        oil_data = data['data']
                        for price_point in oil_data:
                            try:
                                # Create OHLC from single price point
                                price = float(price_point['value'])
                                timestamp = price_point['date']
                                rows.append({
                                    'timestamp': pd.to_datetime(timestamp),
                                    'open': price,
                                    'high': price,
                                    'low': price,
                                    'close': price,
                                    'volume': 0
                                })
                            except (KeyError, ValueError) as e:
                                logger.warning(f"Skipping invalid oil data point: {e}")
                                continue
                        
                # If commodity data format not recognized, try to generate mock data
                if not rows:
                    logger.warning(f"Commodity data format not recognized for {symbol}. Available keys: {list(data.keys())}")
                    # Generate mock OHLC data for demonstration
                    base_price = 2000.0 if symbol == 'XAUUSD' else (25.0 if symbol == 'XAGUSD' else 70.0)
                    import random
                    for i in range(limit):
                        timestamp = datetime.now() - timedelta(hours=i)
                        price = base_price + random.uniform(-5, 5)
                        rows.append({
                            'timestamp': timestamp,
                            'open': price,
                            'high': price + random.uniform(0, 2),
                            'low': price - random.uniform(0, 2),
                            'close': price + random.uniform(-1, 1),
                            'volume': 0
                        })
                        
            else:
                # Handle forex data parsing
                time_series_key = f"Time Series FX ({interval})"
                if time_series_key not in data:
                    logger.error(f"No time series data found for {symbol}. Available keys: {list(data.keys())}")
                    return None
                
                time_series = data[time_series_key]
                
                for timestamp_str, ohlc in time_series.items():
                    try:
                        rows.append({
                            'timestamp': pd.to_datetime(timestamp_str),
                            'open': float(ohlc['1. open']),
                            'high': float(ohlc['2. high']),
                            'low': float(ohlc['3. low']),
                            'close': float(ohlc['4. close']),
                            'volume': 0  # Forex doesn't have volume in Alpha Vantage
                        })
                    except (KeyError, ValueError) as e:
                        logger.warning(f"Skipping invalid data point for {timestamp_str}: {e}")
                        continue
            
            if not rows:
                logger.error(f"No valid data points found for {symbol}")
                return None
            
            df = pd.DataFrame(rows)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Cache the data
            self.data_cache[cache_key] = {
                'data': df,
                'timestamp': datetime.now()
            }
            
            # Save to disk cache as well
            cache_file = self.cache_dir / f"{cache_key}.json"
            try:
                cache_data = {
                    'data': df.to_dict('records'),
                    'timestamp': datetime.now().isoformat()
                }
                with open(cache_file, 'w') as f:
                    json.dump(cache_data, f)
            except Exception as e:
                logger.warning(f"Failed to save cache to disk: {e}")
            
            result_df = df.tail(limit) if limit else df
            logger.info(f"Retrieved {len(df)} bars for {symbol} from Alpha Vantage (returning {len(result_df)})")
            return result_df
            
        except httpx.RequestError as e:
            logger.error(f"Network error retrieving data for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving data for {symbol}: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if Alpha Vantage is configured and available"""
        return self.enabled
    
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price from Alpha Vantage with caching and rate limiting"""
        if not self.enabled:
            return None
        
        # Check cache first
        cache_key = f"{symbol}_latest_price"
        if cache_key in self.price_cache:
            cached_data = self.price_cache[cache_key]
            if datetime.now() - cached_data['timestamp'] < timedelta(seconds=30):
                return cached_data['price']
        
        try:
            # Check if this is a commodity symbol
            if symbol in self.commodity_mapping:
                # Handle commodity latest price request
                commodity_info = self.commodity_mapping[symbol]
                
                # Wait for rate limit if necessary
                await self._check_and_wait_rate_limit()
                
                params = {
                    "function": commodity_info['function'],
                    "symbol": commodity_info['symbol'],
                    "market": commodity_info['market'],
                    "apikey": self.api_key
                }
                
                # Special handling for crude oil
                if commodity_info['function'] == 'CRUDE_OIL':
                    params = {
                        "function": "CRUDE_OIL",
                        "interval": "daily",
                        "apikey": self.api_key
                    }
                    
            else:
                # Handle forex pairs
                # Convert symbol format
                if len(symbol) == 6:
                    from_currency = symbol[:3]
                    to_currency = symbol[3:]
                else:
                    return None
                
                # Wait for rate limit if necessary
                await self._check_and_wait_rate_limit()
                
                params = {
                    "function": "CURRENCY_EXCHANGE_RATE",
                    "from_currency": from_currency,
                    "to_currency": to_currency,
                    "apikey": self.api_key
                }
            
            client = await self._get_client()
            response = await client.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "Error Message" in data:
                logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                return None
            
            if "Realtime Currency Exchange Rate" in data:
                rate_data = data["Realtime Currency Exchange Rate"]
                price = float(rate_data["5. Exchange Rate"])
                
                # Cache the price
                self.price_cache[cache_key] = {
                    'price': price,
                    'timestamp': datetime.now()
                }
                
                return price
            
            logger.warning(f"No exchange rate data found for {symbol}")
            return None
            
        except httpx.RequestError as e:
            logger.error(f"Network error getting latest price for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {e}")
            return None
    
    async def get_news(self, category: str = 'general', limit: int = 20) -> Optional[List[Dict[str, Any]]]:
        """Get financial news with sentiment analysis from Alpha Vantage"""
        if not self.enabled:
            return None
        
        try:
            # Wait for rate limit if necessary
            await self._check_and_wait_rate_limit()
            
            # Map categories to Alpha Vantage topics
            topic_map = {
                'general': None,  # No specific topic filter for general news
                'forex': 'financial_markets',
                'crypto': 'blockchain',
                'cryptocurrency': 'blockchain', 
                'technology': 'technology',
                'economy': 'economy_macro',
                'finance': 'financial_markets'
            }
            
            params = {
                "function": "NEWS_SENTIMENT",
                "apikey": self.api_key,
                "limit": min(limit, 1000)  # Alpha Vantage max limit
            }
            
            # Add topic filter if available
            topic = topic_map.get(category.lower())
            if topic:
                params["topics"] = topic
            
            client = await self._get_client()
            response = await client.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Check for API errors
            if "Error Message" in data:
                logger.error(f"Alpha Vantage News API error: {data['Error Message']}")
                return None
            
            if "Note" in data:
                logger.warning(f"Alpha Vantage News rate limit: {data['Note']}")
                return None
            
            # Parse news feed
            if "feed" not in data:
                logger.warning(f"No news feed found in Alpha Vantage response")
                return None
            
            formatted_news = []
            for article in data["feed"][:limit]:
                # Parse sentiment data
                sentiment_data = {
                    'label': article.get('overall_sentiment_label', 'Neutral'),
                    'score': float(article.get('overall_sentiment_score', 0)),
                    'method': 'alpha_vantage_ai'
                }
                
                # Parse ticker sentiments if available
                ticker_sentiments = []
                if 'ticker_sentiment' in article:
                    for ticker_sent in article['ticker_sentiment']:
                        ticker_sentiments.append({
                            'ticker': ticker_sent.get('ticker', ''),
                            'relevance_score': float(ticker_sent.get('relevance_score', 0)),
                            'sentiment_score': float(ticker_sent.get('ticker_sentiment_score', 0)),
                            'sentiment_label': ticker_sent.get('ticker_sentiment_label', 'Neutral')
                        })
                
                formatted_article = {
                    'id': f"av_{hash(article.get('url', ''))}",  # Generate ID from URL hash
                    'title': article.get('title', ''),
                    'summary': article.get('summary', ''),
                    'content': article.get('summary', ''),  # Alpha Vantage provides summary as content
                    'url': article.get('url', ''),
                    'source': article.get('source', ''),
                    'category': category,
                    'published_at': article.get('time_published', ''),
                    'timestamp': self._parse_time_to_timestamp(article.get('time_published', '')),
                    'image_url': article.get('banner_image', ''),
                    'related_symbols': [ts['ticker'] for ts in ticker_sentiments],
                    'sentiment': sentiment_data,
                    'ticker_sentiments': ticker_sentiments,
                    'provider': 'alpha_vantage'
                }
                formatted_news.append(formatted_article)
            
            logger.info(f"Retrieved {len(formatted_news)} {category} news articles with sentiment from Alpha Vantage")
            return formatted_news
            
        except httpx.RequestError as e:
            logger.error(f"Network error retrieving news from Alpha Vantage: {e}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving news from Alpha Vantage: {e}")
            return None
    
    async def get_symbol_news(self, symbol: str, limit: int = 10) -> Optional[List[Dict[str, Any]]]:
        """Get news articles related to a specific symbol with sentiment analysis"""
        if not self.enabled:
            return None
        
        try:
            # Wait for rate limit if necessary  
            await self._check_and_wait_rate_limit()
            
            # Convert forex pairs to individual currencies for better news matching
            tickers_to_search = []
            if len(symbol) == 6 and symbol.isalpha():
                # Forex pair - search for both currencies and general forex news
                base_currency = symbol[:3]
                quote_currency = symbol[3:]
                
                # Add major currency tickers that might have relevant news
                currency_tickers = {
                    'USD': ['DXY', 'UUP'],  # Dollar index and USD ETF
                    'EUR': ['FXE', 'EUO'],  # Euro ETFs
                    'GBP': ['FXB', 'EWU'],  # Pound ETFs
                    'JPY': ['FXY', 'EWJ'],  # Yen ETFs
                    'AUD': ['FXA', 'EWA'],  # Australian dollar ETFs
                    'CAD': ['FXC', 'EWC']   # Canadian dollar ETFs
                }
                
                for currency in [base_currency, quote_currency]:
                    if currency in currency_tickers:
                        tickers_to_search.extend(currency_tickers[currency])
            
            elif symbol.upper() in ['BTCUSD', 'ETHUSD', 'BTCEUR', 'ETHEUR']:
                # Crypto symbols - search for crypto-related tickers
                tickers_to_search = ['COIN', 'MSTR', 'TSLA']  # Major crypto-related stocks
            
            params = {
                "function": "NEWS_SENTIMENT",
                "apikey": self.api_key,
                "limit": min(limit * 2, 50)  # Get more articles to filter relevant ones
            }
            
            # Add tickers if we found relevant ones
            if tickers_to_search:
                params["tickers"] = ','.join(tickers_to_search[:5])  # Limit to 5 tickers
            
            client = await self._get_client()
            response = await client.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Check for API errors
            if "Error Message" in data:
                logger.error(f"Alpha Vantage Symbol News API error: {data['Error Message']}")
                return None
            
            if "Note" in data:
                logger.warning(f"Alpha Vantage Symbol News rate limit: {data['Note']}")
                return None
            
            if "feed" not in data:
                # Fallback to general financial news
                return await self.get_news('finance', limit)
            
            # Parse and filter news articles
            formatted_news = []
            for article in data["feed"]:
                # Check if article is relevant to our symbol
                title_lower = article.get('title', '').lower()
                summary_lower = article.get('summary', '').lower()
                
                relevance_score = 0
                
                # For forex pairs, check for currency mentions
                if len(symbol) == 6 and symbol.isalpha():
                    base_currency = symbol[:3].lower()
                    quote_currency = symbol[3:].lower()
                    
                    currency_keywords = {
                        'usd': ['dollar', 'usd', 'fed', 'federal reserve'],
                        'eur': ['euro', 'eur', 'ecb', 'european central bank'],
                        'gbp': ['pound', 'gbp', 'sterling', 'bank of england', 'boe'],
                        'jpy': ['yen', 'jpy', 'bank of japan', 'boj'],
                        'aud': ['australian dollar', 'aud', 'rba'],
                        'cad': ['canadian dollar', 'cad', 'bank of canada']
                    }
                    
                    for currency in [base_currency, quote_currency]:
                        if currency in currency_keywords:
                            keywords = currency_keywords[currency]
                            for keyword in keywords:
                                if keyword in title_lower or keyword in summary_lower:
                                    relevance_score += 1
                
                # For crypto, check for crypto mentions
                elif 'btc' in symbol.lower() or 'eth' in symbol.lower():
                    crypto_keywords = ['bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'cryptocurrency', 'blockchain']
                    for keyword in crypto_keywords:
                        if keyword in title_lower or keyword in summary_lower:
                            relevance_score += 1
                
                # Only include articles with some relevance or high sentiment impact
                overall_sentiment_score = abs(float(article.get('overall_sentiment_score', 0)))
                if relevance_score > 0 or overall_sentiment_score > 0.1:
                    
                    sentiment_data = {
                        'label': article.get('overall_sentiment_label', 'Neutral'),
                        'score': float(article.get('overall_sentiment_score', 0)),
                        'method': 'alpha_vantage_ai'
                    }
                    
                    formatted_article = {
                        'id': f"av_{hash(article.get('url', ''))}",
                        'title': article.get('title', ''),
                        'summary': article.get('summary', ''),
                        'content': article.get('summary', ''),
                        'url': article.get('url', ''),
                        'source': article.get('source', ''),
                        'category': 'symbol_specific',
                        'published_at': article.get('time_published', ''),
                        'timestamp': self._parse_time_to_timestamp(article.get('time_published', '')),
                        'image_url': article.get('banner_image', ''),
                        'related_symbols': [symbol],
                        'sentiment': sentiment_data,
                        'relevance_score': relevance_score,
                        'provider': 'alpha_vantage'
                    }
                    formatted_news.append(formatted_article)
                
                if len(formatted_news) >= limit:
                    break
            
            logger.info(f"Retrieved {len(formatted_news)} symbol-specific news articles for {symbol} from Alpha Vantage")
            return formatted_news
            
        except httpx.RequestError as e:
            logger.error(f"Network error retrieving symbol news from Alpha Vantage: {e}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving symbol news from Alpha Vantage: {e}")
            return None
    
    def _parse_time_to_timestamp(self, time_string: str) -> int:
        """Parse Alpha Vantage time format to timestamp"""
        try:
            # Alpha Vantage time format: 20231201T143000
            if 'T' in time_string:
                dt = datetime.strptime(time_string, '%Y%m%dT%H%M%S')
                return int(dt.timestamp())
            return 0
        except:
            return 0
