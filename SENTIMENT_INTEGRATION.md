# Sentiment Analysis Integration

## Overview

The Forex Signal Dashboard now includes advanced sentiment analysis that enhances trading signal confidence by analyzing recent news sentiment for currency pairs and cryptocurrencies.

## Features

✅ **Real-time Sentiment Analysis**: Analyzes sentiment from recent news articles (last 24-48 hours)
✅ **Symbol-specific Analysis**: Maps news to relevant currency pairs (e.g., USD news affects EURUSD, GBPUSD, etc.)
✅ **Confidence Enhancement**: Adjusts signal confidence based on market sentiment
✅ **Configurable Parameters**: Fully configurable via environment variables
✅ **Graceful Fallback**: Continues operation even if sentiment analysis fails
✅ **Database Integration**: Leverages existing NewsArticle and NewsSentiment tables

## How It Works

### Sentiment Impact Logic

- **Positive Sentiment** (>0.3): Boosts signal confidence by up to +10%
- **Negative Sentiment** (<-0.3): Reduces signal confidence by up to -10% 
- **Neutral Sentiment** (-0.3 to 0.3): Minimal impact on confidence
- **Time Decay**: Recent news has more weight than older articles
- **Multi-analyzer Support**: Uses VADER, TextBlob, and financial keyword analysis

### Signal Enhancement Process

1. **Signal Generation**: Strategy generates initial signal with base confidence
2. **Sentiment Analysis**: System queries recent news for relevant symbols
3. **Sentiment Calculation**: Aggregates sentiment scores with time decay
4. **Confidence Adjustment**: Applies sentiment impact to signal confidence
5. **Metadata Storage**: Stores sentiment score, impact, and reasoning in database

### Symbol Mapping

Currency pairs are intelligently mapped to relevant news:

- **EURUSD**: EUR, USD, ECB, FED, EURO, DOLLAR news
- **GBPUSD**: GBP, USD, BOE, FED, POUND, STERLING, DOLLAR news
- **USDJPY**: USD, JPY, FED, BOJ, DOLLAR, YEN news
- **BTCUSD**: BTC, BITCOIN, CRYPTO, CRYPTOCURRENCY news
- And more...

## Configuration

### Environment Variables

```bash
# Core settings
SENTIMENT_FACTOR_ENABLED=true           # Enable sentiment analysis (default: true)
SENTIMENT_WEIGHT=0.1                    # Max impact on confidence (default: 0.1 = 10%)
SENTIMENT_LOOKBACK_HOURS=24             # Hours of news to analyze (default: 24)
SENTIMENT_RECENCY_DECAY=0.5             # Time decay factor (default: 0.5)

# Sentiment thresholds
SENTIMENT_POSITIVE_THRESHOLD=0.3        # Positive sentiment threshold
SENTIMENT_NEGATIVE_THRESHOLD=-0.3       # Negative sentiment threshold  
SENTIMENT_HIGH_CONFIDENCE_THRESHOLD=0.7 # High confidence threshold

# Performance tuning
SENTIMENT_MAX_ARTICLES=50               # Max articles per symbol
SENTIMENT_MIN_ARTICLES=1                # Min articles for analysis

# Reliability
SENTIMENT_GRACEFUL_FALLBACK=true        # Enable graceful fallback
SENTIMENT_TIMEOUT=5.0                   # Analysis timeout in seconds
```

### Example Configuration

```yaml
# docker-compose.yml
environment:
  - SENTIMENT_FACTOR_ENABLED=true
  - SENTIMENT_WEIGHT=0.15              # 15% max impact
  - SENTIMENT_LOOKBACK_HOURS=48        # 2 days of news
```

## Database Schema

### Enhanced Signal Model

New fields added to the `signals` table:

```sql
-- Sentiment analysis fields
sentiment_score FLOAT DEFAULT 0.0,      -- Sentiment score from -1 to 1
sentiment_impact FLOAT DEFAULT 0.0,     -- Impact on confidence
sentiment_reason TEXT,                  -- Human readable explanation
```

### Existing Tables Used

- **news_articles**: Source of news content
- **news_sentiments**: Sentiment analysis results  
- **signals**: Enhanced with sentiment metadata

## API Endpoints

Sentiment data is exposed through existing API endpoints:

- `GET /api/signals/recent` - Includes sentiment fields in signal data
- `GET /api/news/sentiment-summary` - Overall sentiment summary
- `GET /api/news/` - News articles with sentiment analysis

## Logging and Monitoring

### Log Examples

```json
{
  "event": "Sentiment adjusted confidence for EURUSD BUY signal: 0.750 -> 0.825 (impact: +0.075) - POSITIVE",
  "logger": "backend.signals.engine",
  "level": "info"
}
```

### Metrics Tracked

- Sentiment impact on signal confidence
- Number of articles analyzed per symbol
- Sentiment analysis success/failure rates
- Processing time and performance

## Integration Points

### Signal Engine (`backend/signals/engine.py`)

```python
# Sentiment integration happens after signal creation
sentiment_data = await sentiment_factor_service.get_sentiment_factor(symbol, db)
signal.sentiment_score = sentiment_data['sentiment_score']
signal.sentiment_impact = sentiment_data['sentiment_impact'] 
signal.confidence = max(0.0, min(1.0, signal.confidence + sentiment_data['sentiment_impact']))
```

### Sentiment Factor Service (`backend/services/sentiment_factor.py`)

- Analyzes recent news for symbol relevance
- Calculates aggregated sentiment with time decay
- Returns sentiment impact and reasoning
- Handles graceful fallback for reliability

## Performance Considerations

- **Caching**: Sentiment results could be cached for performance
- **Async Processing**: All sentiment analysis is async to avoid blocking
- **Rate Limiting**: Respects database query limits
- **Timeout Protection**: 5-second timeout prevents hanging
- **Graceful Degradation**: Falls back to neutral sentiment if analysis fails

## Testing

### Verification Steps

1. **Service Import**: `python -c "from backend.services.sentiment_factor import sentiment_factor_service; print('Success')"`
2. **Configuration**: Check logs for "Sentiment factor service initialized" message
3. **Signal Enhancement**: Look for sentiment adjustment logs during signal generation
4. **Database**: Verify sentiment fields are populated in signals table

### Example Test

```python
# Test sentiment factor calculation
from backend.services.sentiment_factor import sentiment_factor_service
from backend.database import SessionLocal

db = SessionLocal()
result = await sentiment_factor_service.get_sentiment_factor('EURUSD', db)
print(f"Sentiment impact: {result['sentiment_impact']}")
```

## Future Enhancements

- **Real-time News Streaming**: Integrate with real-time news feeds
- **Sentiment Caching**: Cache sentiment results for performance  
- **Machine Learning**: Train custom sentiment models for financial news
- **Correlation Analysis**: Analyze sentiment vs actual price movements
- **Advanced NLP**: Use transformer models for better sentiment accuracy

## Troubleshooting

### Common Issues

1. **No Sentiment Data**: Check if news collection is running and NewsArticle table has recent data
2. **Performance Issues**: Adjust `SENTIMENT_MAX_ARTICLES` and `SENTIMENT_TIMEOUT`
3. **Disabled Sentiment**: Check `SENTIMENT_FACTOR_ENABLED` environment variable
4. **Import Errors**: Ensure all dependencies are installed (vaderSentiment, textblob)

### Debug Commands

```bash
# Check sentiment service status
python -c "from backend.services.sentiment_factor import sentiment_factor_service; print(sentiment_factor_service.get_configuration())"

# Check recent news data
echo "SELECT COUNT(*) FROM news_articles WHERE published_at > NOW() - INTERVAL '24 HOURS';" | psql $DATABASE_URL

# Check sentiment analysis results  
echo "SELECT COUNT(*) FROM news_sentiments WHERE analyzed_at > NOW() - INTERVAL '24 HOURS';" | psql $DATABASE_URL
```

## Summary

The sentiment analysis integration enhances trading signals by incorporating market sentiment from recent news, providing traders with additional context for their decisions while maintaining system reliability through graceful fallbacks and configurable parameters.