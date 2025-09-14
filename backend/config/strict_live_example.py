"""
STRICT LIVE MODE EXAMPLE CONFIGURATION

This file demonstrates how to properly enable and configure STRICT_LIVE_MODE 
for production deployment.

To enable strict mode in production:

1. Set environment variables before starting the application:
   export STRICT_LIVE_MODE=true
   export STRICT_LIVE_MAX_DATA_AGE=10.0
   export STRICT_LIVE_MIN_PROVIDERS=2
   export STRICT_LIVE_APPROVED_SOURCES="Polygon.io,MT5"

2. Or create a .env file with:
   STRICT_LIVE_MODE=true
   STRICT_LIVE_MAX_DATA_AGE=10.0
   STRICT_LIVE_MIN_PROVIDERS=2
   STRICT_LIVE_APPROVED_SOURCES="Polygon.io,MT5"

3. For Docker deployment, add to docker-compose.yml:
   environment:
     - STRICT_LIVE_MODE=true
     - STRICT_LIVE_MAX_DATA_AGE=10.0
     - STRICT_LIVE_MIN_PROVIDERS=2
     - STRICT_LIVE_APPROVED_SOURCES=Polygon.io,MT5

STRICT MODE VALIDATION CHECKLIST:
✅ No synthetic/mock data allowed
✅ Data must be < 15 seconds old (configurable)
✅ Only approved live data sources
✅ Market must be open (configurable)
✅ Minimum data bars requirement
✅ Real data marker required
✅ Live source validation
✅ Cross-provider verification

EXAMPLE PRODUCTION CONFIGURATION:
- STRICT_LIVE_MODE=true
- STRICT_LIVE_MAX_DATA_AGE=5.0          # Ultra-strict 5-second limit
- STRICT_LIVE_MIN_PROVIDERS=2           # Require 2 provider validations
- STRICT_LIVE_APPROVED_SOURCES="Polygon.io,MT5"  # Only professional sources
- STRICT_LIVE_BLOCKED_SOURCES="ExchangeRate.host,AlphaVantage,MockDataProvider"
- STRICT_LIVE_REQUIRE_MARKET_OPEN=true  # Block signals when market closed
- STRICT_LIVE_VERBOSE_LOGGING=true      # Full audit trail
"""

# TEMPORARY TEST CONFIGURATION
# For immediate testing, uncomment these lines in backend/config/strict_live_config.py:

TEST_CONFIG = """
# Replace this line in StrictLiveConfig:
# ENABLED = os.getenv('STRICT_LIVE_MODE', 'false').lower() == 'true'

# With this line for testing:
# ENABLED = True  # TEMPORARY: Force enable for testing
"""

PRODUCTION_SETUP = """
# Production deployment checklist:
1. Ensure approved data providers are available and configured
2. Set STRICT_LIVE_MODE=true in environment
3. Configure provider API keys and endpoints  
4. Test with development data first
5. Monitor logs for strict mode validation messages
6. Verify signal blocking behavior under strict mode
"""