"""
Signal Generation Engine
"""
import pandas as pd
import os
import time
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from typing import Optional, List, Dict, Tuple, Any

from ..models import Signal, Strategy
# from ..services.sentiment_factor import sentiment_factor_service  # Temporarily disabled
from ..logs.logger import get_logger
from ..ai_capabilities import get_ai_capabilities, OPENAI_ENABLED
from ..services.manus_ai import ManusAI
from ..instruments.metadata import instrument_db, AssetClass, get_instrument_metadata, get_pip_size, format_price
from ..config.strict_live_config import StrictLiveConfig

# Initialize logger first
logger = get_logger(__name__)

# Check enhanced AI capabilities
ai_capabilities = get_ai_capabilities()
CHATGPT_AVAILABLE = ai_capabilities['openai_enabled']
MULTI_AI_AVAILABLE = ai_capabilities['multi_ai_enabled']

# Conditional imports for AI components
ChatGPTStrategyOptimizer = None
AIStrategyConsensus = None
AdvancedBacktester = None
MultiAIConsensus = None

# Try to import Multi-AI Consensus system first
if MULTI_AI_AVAILABLE:
    try:
        from ..services.multi_ai_consensus import MultiAIConsensus
        logger.info(f"Multi-AI Consensus system loaded - {ai_capabilities['total_ai_agents']} agents available")
    except ImportError as e:
        logger.warning(f"Multi-AI Consensus system failed to load: {e}")
        MULTI_AI_AVAILABLE = False

# Legacy ChatGPT-only components
if CHATGPT_AVAILABLE:
    try:
        from ..services.chatgpt_strategy_optimizer import ChatGPTStrategyOptimizer
        from ..services.ai_strategy_consensus import AIStrategyConsensus
        from ..services.advanced_backtester import AdvancedBacktester
        logger.info("ChatGPT integration loaded successfully - dual-AI mode enabled")
    except ImportError as e:
        logger.warning(f"ChatGPT integration failed to load: {e}")
        CHATGPT_AVAILABLE = False
else:
    logger.info("ChatGPT integration not available - using enhanced AI system")

# Always try to load advanced backtester
try:
    if not AdvancedBacktester:
        from ..services.advanced_backtester import AdvancedBacktester
        logger.info("Advanced backtester loaded")
except ImportError as e:
    logger.warning(f"Advanced backtester not available: {e}")
from ..providers.mock import MockDataProvider
from ..providers.alphavantage import AlphaVantageProvider
from ..providers.freecurrency import FreeCurrencyAPIProvider
from ..providers.mt5_data import MT5DataProvider
from ..providers.finnhub_provider import FinnhubProvider
from ..providers.exchangerate_provider import ExchangeRateProvider
from ..providers.polygon_provider import PolygonProvider
from ..providers.coingecko_provider import CoinGeckoProvider
from ..providers.binance_provider import BinanceProvider
from ..providers.coinbase_provider import CoinbaseProvider
from ..risk.guards import RiskManager
from ..regime.detector import regime_detector
from ..providers.execution.mt5_bridge import MT5BridgeExecutionProvider
from ..providers.execution.base import OrderRequest, OrderType
from .strategies.ema_rsi import EMAStragey
from .strategies.donchian_atr import DonchianATRStrategy
from .strategies.meanrev_bb import MeanReversionBBStrategy
from .strategies.macd_strategy import MACDStrategy
from .strategies.stochastic_strategy import StochasticStrategy
from .strategies.rsi_divergence import RSIDivergenceStrategy
from .strategies.fibonacci_strategy import FibonacciStrategy

def is_forex_market_open() -> bool:
    """
    Check if Forex market is currently open
    Forex markets operate Monday 00:00 UTC to Friday 22:00 UTC
    """
    now = datetime.utcnow()
    weekday = now.weekday()  # Monday = 0, Sunday = 6
    
    # Market closed on weekends (Saturday = 5, Sunday = 6)
    if weekday == 5:  # Saturday
        return False
    elif weekday == 6:  # Sunday
        return False
    elif weekday == 0:  # Monday - open from 00:00 UTC
        return True
    elif weekday == 4:  # Friday - close at 22:00 UTC
        return now.hour < 22
    else:  # Tuesday, Wednesday, Thursday - fully open
        return True

class SignalEngine:
    """Main signal generation engine"""
    
    def __init__(self):
        # Initialize data providers (priority order: Coinbase -> Binance -> CoinGecko -> Polygon.io -> ExchangeRate.host -> Finnhub -> FreeCurrency -> Alpha Vantage -> Mock)
        self.coinbase_provider = CoinbaseProvider()
        self.binance_provider = BinanceProvider()
        self.coingecko_provider = CoinGeckoProvider()
        self.polygon_provider = PolygonProvider()
        self.exchangerate_provider = ExchangeRateProvider()
        self.finnhub_provider = FinnhubProvider()
        self.mt5_provider = MT5DataProvider()
        self.freecurrency_provider = FreeCurrencyAPIProvider()
        self.alphavantage_provider = AlphaVantageProvider()
        self.mock_provider = MockDataProvider()
        
        # Strategy mapping - 7 Advanced Trading Strategies
        self.strategies = {
            'ema_rsi': EMAStragey(),
            'donchian_atr': DonchianATRStrategy(),
            'meanrev_bb': MeanReversionBBStrategy(),
            'macd_crossover': MACDStrategy(),
            'stochastic': StochasticStrategy(),
            'rsi_divergence': RSIDivergenceStrategy(),
            'fibonacci': FibonacciStrategy()
        }
        
        self.execution_provider = MT5BridgeExecutionProvider()
        
        # Initialize enhanced AI system for collaborative trading recommendations
        self.manus_ai = ManusAI()
        
        # Initialize Multi-AI Consensus system if available
        if MULTI_AI_AVAILABLE and MultiAIConsensus:
            try:
                self.multi_ai_consensus = MultiAIConsensus()
                self.enable_multi_ai = True
                logger.info("SignalEngine: Multi-AI Consensus system initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Multi-AI Consensus: {e}")
                self.enable_multi_ai = False
                self.multi_ai_consensus = None
        else:
            self.enable_multi_ai = False
            self.multi_ai_consensus = None
            
        # Initialize legacy ChatGPT components if available
        if CHATGPT_AVAILABLE and ChatGPTStrategyOptimizer and AIStrategyConsensus:
            try:
                self.chatgpt_optimizer = ChatGPTStrategyOptimizer()
                self.ai_consensus = AIStrategyConsensus()
                logger.info("SignalEngine: Legacy ChatGPT components initialized successfully")
            except Exception as e:
                logger.warning(f"SignalEngine: Failed to initialize ChatGPT components: {e}")
                self.chatgpt_optimizer = None
                self.ai_consensus = None
        else:
            self.chatgpt_optimizer = None
            self.ai_consensus = None
            logger.info("SignalEngine: ChatGPT components not available, using Manus AI only")
        
        # Initialize AdvancedBacktester conditionally
        if AdvancedBacktester:
            try:
                self.advanced_backtester = AdvancedBacktester()
                logger.info("SignalEngine: Advanced backtester initialized successfully")
            except Exception as e:
                logger.warning(f"SignalEngine: Failed to initialize advanced backtester: {e}")
                self.advanced_backtester = None
        else:
            self.advanced_backtester = None
            logger.warning("SignalEngine: Advanced backtester not available")
        
        # AI system configuration
        self.enable_dual_ai = os.getenv('ENABLE_DUAL_AI', 'true').lower() == 'true'
        self.ai_consensus_threshold = float(os.getenv('AI_CONSENSUS_THRESHOLD', '0.4'))  # Lowered from 0.7 to 0.4 for better crypto signal generation
        self.enable_ai_backtesting = os.getenv('ENABLE_AI_BACKTESTING', 'true').lower() == 'true'
        
        # Auto-trading configuration
        self.auto_trade_enabled = os.getenv('AUTO_TRADE_ENABLED', 'false').lower() == 'true'
        self.confidence_threshold = float(os.getenv('AUTO_TRADE_CONFIDENCE_THRESHOLD', '0.85'))
        self.default_lot_size = float(os.getenv('AUTO_TRADE_LOT_SIZE', '0.01'))  # Micro lot
        
        # Strict Live Mode Configuration - Enterprise-grade production safety
        StrictLiveConfig.log_configuration(logger)
    
    def _get_asset_class(self, symbol: str) -> str:
        """Determine asset class for intelligent provider routing using instrument metadata"""
        asset_class = instrument_db.get_asset_class(symbol)
        
        # Map AssetClass enum to string for backward compatibility
        if asset_class == AssetClass.CRYPTO:
            return 'crypto'
        elif asset_class in [AssetClass.METALS, AssetClass.OIL]:
            return 'metals_oil'
        elif asset_class == AssetClass.FOREX:
            return 'forex'
        else:
            # Fallback for unknown symbols
            return 'forex'
    
    def _get_providers_for_asset_class(self, asset_class: str) -> List[tuple]:
        """Get prioritized list of providers for specific asset class"""
        # Provider tuples: (provider_instance, provider_name)
        if asset_class == 'crypto':
            # Crypto: Use providers that support crypto symbols (Coinbase primary for real OHLC data)
            return [
                (self.coinbase_provider, 'Coinbase'),  # Primary crypto provider with real OHLC data
                (self.binance_provider, 'Binance'),  # Secondary (may be geo-blocked) 
                (self.coingecko_provider, 'CoinGecko'),  # Tertiary crypto API 
                (self.exchangerate_provider, 'ExchangeRate.host'),  # Has crypto hardcoded rates
                (self.mt5_provider, 'MT5'),  # May support crypto
            ]
        elif asset_class == 'metals_oil':
            # Metals/Oil: Use providers that support commodity symbols  
            return [
                (self.mt5_provider, 'MT5'),  # Professional metals/oil data
                (self.exchangerate_provider, 'ExchangeRate.host'),  # Has metals hardcoded rates
                (self.polygon_provider, 'Polygon.io'),  # May support some commodities
            ]
        else:  # forex
            # Forex: All providers support forex, prioritize by quality
            return [
                (self.polygon_provider, 'Polygon.io'),  # Professional forex data
                (self.mt5_provider, 'MT5'),  # Professional trading platform
                (self.exchangerate_provider, 'ExchangeRate.host'),  # Free unlimited forex
                (self.finnhub_provider, 'Finnhub'),  # Real-time forex (OANDA)
                (self.freecurrency_provider, 'FreeCurrency'),  # Live forex rates
                (self.alphavantage_provider, 'AlphaVantage'),  # Traditional forex API
            ]
    
    def _is_market_open_for_symbol(self, symbol: str) -> bool:
        """Check if market is open for the specific symbol type using instrument metadata"""
        instrument = get_instrument_metadata(symbol)
        
        if instrument is None:
            # Fallback for unknown symbols - assume forex market hours
            logger.warning(f"Unknown instrument {symbol}, using forex market hours as fallback")
            return is_forex_market_open()
        
        # Check if market is 24/7 (crypto)
        if instrument.is_24_7:
            logger.debug(f"Market for {symbol} is always open (24/7)")
            return True
        
        # Check market hours for other instruments
        now = datetime.utcnow()
        weekday = now.weekday()  # Monday = 0, Sunday = 6
        current_hour = now.hour
        
        # Check if today is a trading day
        if weekday not in instrument.market_open_days:
            logger.debug(f"Market for {symbol} is closed today (weekday {weekday})")
            return False
        
        # Check trading hours
        start_hour, end_hour = instrument.market_open_hours
        if start_hour <= current_hour < end_hour:
            logger.debug(f"Market for {symbol} is open (hour {current_hour} within {start_hour}-{end_hour})")
            return True
        else:
            logger.debug(f"Market for {symbol} is closed (hour {current_hour} outside {start_hour}-{end_hour})")
            return False
    
    async def process_symbol(self, symbol: str, db: Session):
        """Process signals for a single symbol"""
        try:
            # STRICT MODE: Enhanced market open validation
            if not self._is_market_open_for_symbol(symbol):
                if StrictLiveConfig.ENABLED and StrictLiveConfig.REQUIRE_MARKET_OPEN:
                    logger.warning(f"🔒 STRICT MODE BLOCKED {symbol}: Market closed - production safety requires open market")
                else:
                    logger.info(f"Market closed - skipping signal generation for {symbol}")
                return
                
            logger.debug(f"Processing signals for {symbol}")
            
            # STRICT MODE: Enhanced data retrieval with production safety
            data = await self._get_market_data(symbol)
            if data is None:
                if StrictLiveConfig.ENABLED:
                    logger.error(f"🔒 STRICT MODE BLOCKED {symbol}: No real market data available - signal generation BLOCKED for safety")
                    logger.error(f"🔒 Production safety requires verified real-time data. Ensure approved data providers are available.")
                else:
                    logger.warning(f"No data available for {symbol}")
                return
            
            # Add a new bar to simulate real-time updates
            if hasattr(self.mock_provider, 'add_new_bar'):
                self.mock_provider.add_new_bar(symbol)
                # Refresh data with new bar
                data = await self._get_market_data(symbol)
                
            # Check data sufficiency after refresh
            if data is None or len(data) < 30:
                logger.warning(f"Insufficient data for {symbol}: {len(data) if data is not None else 0} bars")
                return
            
            # Get strategy configurations for this symbol
            strategies = db.query(Strategy).filter(
                Strategy.symbol == symbol,
                Strategy.enabled == True
            ).all()
            
            if not strategies:
                logger.info(f"No enabled strategies for {symbol}")
                return
            
            logger.info(f"Found {len(strategies)} enabled strategies for {symbol}: {[s.name for s in strategies]}")
            
            # Detect market regime first
            regime_data = regime_detector.detect_regime(data, symbol)
            if regime_data['regime'] != 'UNKNOWN':
                regime_detector.store_regime(symbol, regime_data, db)
                logger.debug(f"Market regime for {symbol}: {regime_data['regime']} ({regime_data['confidence']:.2f})")
            
            # Get AI strategy recommendations using enhanced multi-AI system
            if self.enable_multi_ai and self.multi_ai_consensus:
                consensus_recommendations = await self._get_multi_ai_recommendations(symbol, data, [str(s.name) for s in strategies])
            elif self.enable_dual_ai and CHATGPT_AVAILABLE:
                # Fallback to Manus AI for legacy ChatGPT mode
                consensus_recommendations = await self._get_manus_ai_recommendations(symbol, data)
            else:
                # Fallback to single Manus AI
                consensus_recommendations = await self._get_manus_ai_recommendations(symbol, data)
            
            # Process each strategy
            for strategy_config in strategies:
                # Check if strategy is suitable for current regime
                if not regime_detector.is_regime_suitable_for_strategy(regime_data['regime'], str(strategy_config.name)):
                    logger.debug(f"Strategy {strategy_config.name} skipped for {symbol} - unsuitable for {regime_data['regime']} regime")
                    continue
                
                # CRITICAL: Check AI strategy guardrails - actively block "avoid" strategies
                if CHATGPT_AVAILABLE:
                    should_block, block_reason = self._should_block_strategy_ai_consensus(str(strategy_config.name), consensus_recommendations)
                else:
                    should_block, block_reason = self._should_block_strategy(str(strategy_config.name), consensus_recommendations)
                if should_block:
                    logger.warning(f"Strategy {strategy_config.name} BLOCKED for {symbol}: {block_reason}")
                    continue
                
                # Check AI recommendations for strategy prioritization
                if CHATGPT_AVAILABLE:
                    should_prioritize, ai_reason = self._should_prioritize_strategy_ai_consensus(str(strategy_config.name), consensus_recommendations)
                else:
                    should_prioritize, ai_reason = self._should_prioritize_strategy(str(strategy_config.name), consensus_recommendations)
                if should_prioritize:
                    logger.info(f"Strategy {strategy_config.name} prioritized for {symbol}: {ai_reason}")
                
                # Process strategy with enhanced AI analysis
                if self.enable_multi_ai and self.multi_ai_consensus:
                    await self._process_strategy_with_multi_ai(symbol, data, strategy_config, db, consensus_recommendations)
                elif CHATGPT_AVAILABLE:
                    await self._process_strategy_with_ai_consensus(symbol, data, strategy_config, db, consensus_recommendations)
                else:
                    await self._process_strategy(symbol, data, strategy_config, db, consensus_recommendations)
                
        except Exception as e:
            logger.error(f"Error processing symbol {symbol}: {e}")
    
    def _validate_real_data(self, data: pd.DataFrame, symbol: str) -> bool:
        """STRICT validation to ensure only fresh real-time market data is used"""
        if data is None or data.empty:
            if StrictLiveConfig.VERBOSE_LOGGING:
                logger.warning(f"Data validation failed for {symbol}: No data or empty dataset")
            return False
            
        # Check for synthetic data markers in attributes
        if hasattr(data, 'attrs'):
            # STRICT MODE: Apply zero-tolerance checks for synthetic/mock data
            if StrictLiveConfig.ENABLED and StrictLiveConfig.BLOCK_SYNTHETIC_DATA:
                if data.attrs.get('is_synthetic', False):
                    logger.error(f"🔒 STRICT MODE BLOCKED {symbol}: Synthetic data detected")
                    return False
            elif data.attrs.get('is_synthetic', False):
                logger.warning(f"Data validation failed for {symbol}: Contains synthetic data markers")
                return False
                
            if StrictLiveConfig.ENABLED and StrictLiveConfig.BLOCK_MOCK_DATA:
                if data.attrs.get('is_mock', False):
                    logger.error(f"🔒 STRICT MODE BLOCKED {symbol}: Mock data detected")
                    return False
            elif data.attrs.get('is_mock', False):
                logger.warning(f"Data validation failed for {symbol}: Contains mock data markers")
                return False
                
            # STRICT MODE: Require explicit real data marker
            if StrictLiveConfig.ENABLED and StrictLiveConfig.REQUIRE_REAL_DATA_MARKER:
                if not data.attrs.get('is_real_data', False):
                    logger.error(f"🔒 STRICT MODE BLOCKED {symbol}: No real data marker found")
                    return False
            elif not data.attrs.get('is_real_data', False):
                logger.warning(f"Data validation failed for {symbol}: No real data marker found")
                return False
                
            # STRICT MODE: Enhanced data freshness validation
            last_updated = data.attrs.get('last_updated')
            fetch_timestamp = data.attrs.get('fetch_timestamp')
            
            # Use fetch_timestamp (when data was retrieved) if available, otherwise last_updated
            validation_timestamp = fetch_timestamp or last_updated
            
            if validation_timestamp:
                try:
                    from datetime import datetime, timezone
                    # Handle various timestamp formats
                    if isinstance(validation_timestamp, (int, float)):
                        # Unix timestamp
                        update_time = datetime.fromtimestamp(validation_timestamp, tz=timezone.utc)
                    else:
                        # ISO format string
                        update_time = datetime.fromisoformat(str(validation_timestamp).replace('Z', '+00:00'))
                    
                    # Calculate data age with high precision
                    current_time = datetime.now(timezone.utc)
                    time_diff = (current_time - update_time).total_seconds()
                    
                    # STRICT MODE: Apply configurable data age limits
                    max_age = StrictLiveConfig.MAX_DATA_AGE_SECONDS if StrictLiveConfig.ENABLED else 15.0
                    is_fresh, freshness_reason = StrictLiveConfig.validate_data_freshness(time_diff)
                    
                    if is_fresh:
                        if StrictLiveConfig.VERBOSE_LOGGING:
                            logger.info(f"Data freshness PASSED for {symbol}: {freshness_reason}")
                    else:
                        if StrictLiveConfig.ENABLED:
                            logger.error(f"🔒 STRICT MODE BLOCKED {symbol}: {freshness_reason}")
                        else:
                            logger.error(f"Data freshness FAILED for {symbol}: {freshness_reason}")
                        return False
                        
                except Exception as e:
                    logger.error(f"Data validation failed for {symbol}: Could not parse timestamp '{validation_timestamp}': {e}")
                    return False
            else:
                logger.error(f"Data validation failed for {symbol}: No timestamp information available for freshness check")
                return False
            
            # STRICT MODE: Enhanced data source validation
            data_source = data.attrs.get('data_source', 'Unknown')
            is_live_source = data.attrs.get('is_live_source', False)
            
            # STRICT MODE: Check if data source is approved
            if StrictLiveConfig.ENABLED:
                if StrictLiveConfig.is_data_source_blocked(data_source):
                    logger.error(f"🔒 STRICT MODE BLOCKED {symbol}: Data source '{data_source}' is explicitly blocked")
                    return False
                
                if not StrictLiveConfig.is_data_source_approved(data_source):
                    logger.error(f"🔒 STRICT MODE BLOCKED {symbol}: Data source '{data_source}' not in approved list")
                    return False
                
                if StrictLiveConfig.REQUIRE_LIVE_SOURCE and not is_live_source:
                    logger.error(f"🔒 STRICT MODE BLOCKED {symbol}: Live source required but '{data_source}' is not live")
                    return False
                
                if StrictLiveConfig.BLOCK_CACHED_DATA and data_source in ['ExchangeRate.host', 'AlphaVantage']:
                    logger.error(f"🔒 STRICT MODE BLOCKED {symbol}: Cached data source '{data_source}' blocked")
                    return False
                
                if StrictLiveConfig.VERBOSE_LOGGING:
                    logger.info(f"🔒 STRICT MODE APPROVED {symbol}: Data source '{data_source}' validation passed")
            else:
                # Legacy validation for non-strict mode
                verified_live_sources = ['Polygon.io', 'Finnhub', 'MT5', 'FreeCurrencyAPI', 'CoinGecko']
                cached_sources = ['ExchangeRate.host', 'AlphaVantage']  # These may have cached data
                
                if data_source in verified_live_sources and is_live_source:
                    logger.debug(f"Data source verification PASSED for {symbol}: Verified live source '{data_source}'")
                elif data_source in cached_sources:
                    logger.warning(f"Data source verification WARNING for {symbol}: Cached source '{data_source}' - data may not be real-time")
                    # Still allow but with warning for cached sources if timestamp is fresh
                else:
                    logger.error(f"Data validation failed for {symbol}: Unverified data source '{data_source}'")
                    return False
                
        else:
            logger.error(f"Data validation failed for {symbol}: No data attributes found for validation")
            return False
            
        # STRICT MODE: Enhanced data structure and quality validation
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            if StrictLiveConfig.ENABLED:
                logger.error(f"🔒 STRICT MODE BLOCKED {symbol}: Missing required columns {missing_columns}")
            else:
                logger.warning(f"Data validation failed for {symbol}: Missing required columns {missing_columns}")
            return False
        
        # STRICT MODE: Check minimum data bars requirement
        if StrictLiveConfig.ENABLED and len(data) < StrictLiveConfig.MIN_DATA_BARS:
            logger.error(f"🔒 STRICT MODE BLOCKED {symbol}: Insufficient data bars ({len(data)} < {StrictLiveConfig.MIN_DATA_BARS})")
            return False
        elif len(data) < 30:  # Legacy minimum for non-strict mode
            logger.warning(f"Data validation failed for {symbol}: Insufficient data bars ({len(data)} < 30)")
            return False
            
        # Check for reasonable price values (no zeros, negatives, or extreme outliers)
        for col in required_columns:
            if (data[col] <= 0).any():
                if StrictLiveConfig.ENABLED:
                    logger.error(f"🔒 STRICT MODE BLOCKED {symbol}: Invalid {col} values (zero or negative)")
                else:
                    logger.warning(f"Data validation failed for {symbol}: Invalid {col} values (zero or negative)")
                return False
                
        # ENHANCED: Validate latest bar timestamp for real-time requirements
        if hasattr(data.index, 'max'):
            latest_bar_time = data.index.max()
            if pd.notna(latest_bar_time):
                try:
                    # Convert to UTC timezone-aware datetime
                    if latest_bar_time.tz is None:
                        latest_bar_time = latest_bar_time.tz_localize('UTC')
                    
                    bar_age = (datetime.now(timezone.utc) - latest_bar_time).total_seconds()
                    if bar_age > 300:  # Latest bar should be within 5 minutes
                        logger.warning(f"Data validation WARNING for {symbol}: Latest bar is {bar_age:.0f}s old")
                    else:
                        logger.debug(f"Latest bar age for {symbol}: {bar_age:.0f}s")
                except Exception as e:
                    logger.debug(f"Could not validate latest bar timestamp for {symbol}: {e}")
        
        data_source = data.attrs.get('data_source', 'Unknown')
        
        # STRICT MODE: Final validation summary
        if StrictLiveConfig.ENABLED and StrictLiveConfig.VERBOSE_LOGGING:
            logger.info(f"🔒 STRICT MODE VALIDATION PASSED for {symbol}: Production-grade data from {data_source}")
        else:
            logger.info(f"Data validation PASSED for {symbol}: Fresh real-time data from {data_source}")
        return True

    async def _get_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get ONLY real market data from available providers with cross-provider validation"""
        # Determine asset class for intelligent provider routing
        asset_class = self._get_asset_class(symbol)
        providers = self._get_providers_for_asset_class(asset_class)
        
        logger.info(f"Getting {asset_class} data for {symbol} using {len(providers)} compatible providers")
        
        # Track attempts for cross-provider validation
        validation_attempts = []
        successful_data = None
        
        # Try providers in priority order for this asset class
        for provider_instance, provider_name in providers:
            if not provider_instance.is_available():
                logger.debug(f"{provider_name} not available for {symbol}")
                continue
                
            try:
                # Special handling for Polygon.io timeframe conversion
                if provider_name == 'Polygon.io':
                    timeframe_mapping = {'1H': 'H1', '4H': 'H4', '1D': 'D1', '1M': 'M1', '5M': 'M5'}
                    converted_tf = timeframe_mapping.get("1H", 'H1')
                    data = await provider_instance.get_ohlc_data(symbol, timeframe=converted_tf, limit=200)
                else:
                    data = await provider_instance.get_ohlc_data(symbol, limit=200)
                
                # Track validation attempts
                validation_attempt = {
                    'provider': provider_name,
                    'data_available': data is not None,
                    'validation_passed': False,
                    'data_age_seconds': None,
                    'is_live_source': getattr(provider_instance, 'is_live_source', False)
                }
                
                if data is not None:
                    # Extract data age for logging
                    if hasattr(data, 'attrs'):
                        fetch_timestamp = data.attrs.get('fetch_timestamp')
                        if fetch_timestamp:
                            validation_attempt['data_age_seconds'] = time.time() - fetch_timestamp
                    
                    # Validate the data
                    if self._validate_real_data(data, symbol):
                        validation_attempt['validation_passed'] = True
                        successful_data = data
                        validation_attempts.append(validation_attempt)
                        
                        # Use first successful provider (priority order)
                        logger.info(f"Using {provider_name} real {asset_class} data for {symbol}")
                        self._log_cross_provider_validation(symbol, validation_attempts)
                        return data
                    else:
                        logger.warning(f"Rejected {provider_name} data for {symbol}: Failed real data validation")
                else:
                    logger.debug(f"{provider_name} returned no data for {symbol}")
                    
                validation_attempts.append(validation_attempt)
                    
            except Exception as e:
                logger.warning(f"{provider_name} failed for {symbol}: {e}")
                validation_attempts.append({
                    'provider': provider_name,
                    'data_available': False,
                    'validation_passed': False,
                    'error': str(e),
                    'is_live_source': getattr(provider_instance, 'is_live_source', False)
                })
                continue
        
        # Log comprehensive validation summary
        self._log_cross_provider_validation(symbol, validation_attempts)
        
        # CRITICAL: NO FALLBACK TO MOCK/SYNTHETIC DATA FOR LIVE TRADING
        # All data must be real market data or signal generation is blocked
        logger.error(f"CRITICAL: No real {asset_class} data available for {symbol} - Signal generation BLOCKED for safety")
        logger.error(f"Trading signals require real market data. Synthetic/mock data is NOT safe for live trading.")
        logger.error(f"Tried {len(providers)} compatible providers for {asset_class} asset class")
        return None
    
    def _log_cross_provider_validation(self, symbol: str, validation_attempts: List[Dict]):
        """Log detailed cross-provider validation results"""
        logger.info(f"=== CROSS-PROVIDER VALIDATION SUMMARY for {symbol} ===")
        
        live_sources = [v for v in validation_attempts if v.get('is_live_source', False)]
        cached_sources = [v for v in validation_attempts if not v.get('is_live_source', False)]
        successful_attempts = [v for v in validation_attempts if v.get('validation_passed', False)]
        
        logger.info(f"Total providers tested: {len(validation_attempts)}")
        logger.info(f"Live data sources: {len(live_sources)} | Cached sources: {len(cached_sources)}")
        logger.info(f"Successful validations: {len(successful_attempts)}")
        
        for attempt in validation_attempts:
            provider = attempt['provider']
            status = "✅ PASSED" if attempt['validation_passed'] else "❌ FAILED"
            source_type = "LIVE" if attempt.get('is_live_source', False) else "CACHED"
            
            age_info = ""
            if attempt.get('data_age_seconds') is not None:
                age_info = f" (age: {attempt['data_age_seconds']:.1f}s)"
            
            error_info = ""
            if attempt.get('error'):
                error_info = f" - Error: {attempt['error'][:50]}..."
            
            logger.info(f"  {provider} [{source_type}]: {status}{age_info}{error_info}")
        
        if successful_attempts:
            best_attempt = successful_attempts[0]  # First successful (highest priority)
            logger.info(f"✅ SELECTED: {best_attempt['provider']} - Real-time validation PASSED")
        else:
            logger.error(f"❌ NO VALID DATA: All providers failed real-time validation for {symbol}")
        
        logger.info("=== END VALIDATION SUMMARY ===")
    
    async def _process_strategy(
        self, 
        symbol: str, 
        data: pd.DataFrame, 
        strategy_config: Strategy, 
        db: Session,
        manus_recommendations: Optional[Dict] = None
    ):
        """Process a single strategy for a symbol"""
        try:
            strategy_name = strategy_config.name
            if strategy_name not in self.strategies:
                logger.error(f"Unknown strategy: {strategy_name}")
                return
            
            strategy = self.strategies[strategy_name]
            
            # Generate signal
            signal_data = strategy.generate_signal(data, strategy_config.config)
            
            if signal_data is None:
                logger.info(f"No signal generated for {symbol} using {strategy_name}")
                return
            
            # Check for signal deduplication and conflicts
            if self._is_duplicate_signal(symbol, signal_data, strategy_name, db):
                logger.info(f"Duplicate/conflicting signal filtered for {symbol} using {strategy_name}")
                return
                
            # Check cross-strategy consensus
            if not self._get_cross_strategy_consensus(symbol, signal_data, db):
                logger.info(f"Cross-strategy consensus check failed for {symbol} using {strategy_name}")
                return
            
            # Create signal object
            expires_at = datetime.utcnow() + timedelta(
                minutes=strategy_config.config.get('expiry_bars', 60)
            )
            
            # Convert numpy types to Python types for database compatibility
            price = float(signal_data['price']) if signal_data['price'] is not None else None
            sl = float(signal_data.get('sl')) if signal_data.get('sl') is not None else None
            tp = float(signal_data.get('tp')) if signal_data.get('tp') is not None else None
            confidence = float(signal_data['confidence']) if signal_data['confidence'] is not None else None
            
            # Apply Manus AI confidence adjustments
            if manus_recommendations and confidence is not None:
                original_confidence = confidence
                confidence = self._apply_manus_ai_confidence_adjustment(
                    confidence, strategy_name, manus_recommendations
                )
                if abs(confidence - original_confidence) > 0.01:  # Log significant changes
                    logger.info(f"Manus AI adjusted confidence for {symbol} {strategy_name}: "
                               f"{original_confidence:.1%} -> {confidence:.1%}")
            
            signal = Signal(
                symbol=symbol,
                action=signal_data['action'],
                price=price,
                sl=sl,
                tp=tp,
                confidence=confidence,
                strategy=strategy_name,
                expires_at=expires_at
            )
            
            # Apply sentiment analysis to enhance signal confidence - TEMPORARILY DISABLED
            # try:
            #     sentiment_data = await sentiment_factor_service.get_sentiment_factor(symbol, db)
            #     
            #     # Store sentiment information in signal
            #     signal.sentiment_score = sentiment_data['sentiment_score']
            #     signal.sentiment_impact = sentiment_data['sentiment_impact']
            #     signal.sentiment_reason = sentiment_data['reasoning']
            #     
            #     # Apply sentiment impact to confidence (with bounds checking)
            #     original_confidence = signal.confidence
            #     adjusted_confidence = signal.confidence + sentiment_data['sentiment_impact']
            #     signal.confidence = max(0.0, min(1.0, adjusted_confidence))  # Clamp between 0 and 1
            #     
            #     # Log sentiment impact
            #     if sentiment_data['sentiment_impact'] != 0:
            #         logger.info(f"Sentiment adjusted confidence for {symbol} {signal.action} signal: "
            #                   f"{original_confidence:.3f} -> {signal.confidence:.3f} "
            #                   f"(impact: {sentiment_data['sentiment_impact']:+.3f}) - {sentiment_data['sentiment_label']}")
            #     
            # except Exception as e:
            #     logger.warning(f"Failed to apply sentiment factor for {symbol}: {e}")
            #     # Set default sentiment values if analysis fails
            #     signal.sentiment_score = 0.0
            #     signal.sentiment_impact = 0.0
            #     signal.sentiment_reason = f"Sentiment analysis failed: {str(e)}"
            
            # Apply risk management
            risk_manager = RiskManager(db)
            risk_check = risk_manager.check_signal(signal, data)
            
            if not risk_check['allowed']:
                signal.blocked_by_risk = True  # type: ignore[assignment]
                signal.risk_reason = risk_check['reason']  # type: ignore[assignment]
                logger.info(f"Signal blocked by risk management: {risk_check['reason']}")
            else:
                
                # Automatic trade execution if enabled and confidence threshold met
                if self.auto_trade_enabled and signal.confidence >= self.confidence_threshold:  # type: ignore[operator]
                    try:
                        await self._execute_auto_trade(signal, db)
                    except Exception as e:
                        logger.error(f"Auto-trade execution failed for signal {signal.id}: {e}")
            
            # Save signal to database
            db.add(signal)
            db.commit()
            
            logger.info(f"Signal created for {symbol}: {signal_data['action']} @ {signal_data['price']}")
            
        except Exception as e:
            logger.error(f"Error processing strategy {strategy_config.name} for {symbol}: {e}")
            db.rollback()
    
    async def _process_strategy_with_multi_ai(
        self, 
        symbol: str, 
        data: pd.DataFrame, 
        strategy_config: Strategy, 
        db: Session,
        consensus_recommendations: Optional[Dict] = None
    ):
        """Process a single strategy for a symbol with multi-AI consensus enhancement"""
        try:
            strategy_name = strategy_config.name
            if strategy_name not in self.strategies:
                logger.error(f"Unknown strategy: {strategy_name}")
                return
            
            strategy = self.strategies[strategy_name]
            
            # Generate base signal
            base_signal = strategy.generate_signal(data, strategy_config.config)
            
            if base_signal is None:
                logger.info(f"No signal generated for {symbol} using {strategy_name}")
                return
            
            # Early deduplication and conflict checks (before expensive AI calls)
            if self._is_duplicate_signal(symbol, base_signal, strategy_name, db):
                logger.info(f"Duplicate/conflicting signal filtered for {symbol} using {strategy_name}")
                return
                
            # Check cross-strategy consensus
            if not self._get_cross_strategy_consensus(symbol, base_signal, db):
                logger.info(f"Cross-strategy consensus check failed for {symbol} using {strategy_name}")
                return
            
            # Run multi-AI consensus analysis
            if self.multi_ai_consensus is not None:
                consensus = await self.multi_ai_consensus.generate_enhanced_signal_analysis(symbol, data, base_signal)
                
                # **CRITICAL**: Enforce multi-AI quality gates
                if not consensus.get('multi_ai_valid', False):
                    logger.warning(f"Multi-AI consensus failed quality gates for {symbol}: {consensus.get('quality_gate', 'unknown')}")
                    return  # Block signal generation
                    
                # Additional validation: never allow signals with 0 agents
                participating_agents = consensus.get('participating_agents', 0)
                if participating_agents == 0:
                    logger.error(f"CRITICAL: Multi-AI consensus returned 0 participating agents for {symbol} - blocking signal")
                    return
                    
                logger.info(f"Multi-AI consensus passed quality gates for {symbol}: {participating_agents} agents, consensus: {consensus.get('consensus_level', 0):.1%}")
            else:
                # Fallback when multi-AI is not available
                consensus = {
                    'final_confidence': base_signal.get('confidence', 0.5), 
                    'consensus_level': 0.0,
                    'participating_agents': 0,
                    'multi_ai_valid': False
                }
                logger.warning(f"Multi-AI consensus not available for {symbol} - using fallback processing")
            
            # Derive final action and confidence from multi-AI consensus
            final_action = base_signal['action']
            consensus_action = consensus.get('final_action')
            consensus_level = consensus.get('consensus_level', 0.0)
            ai_consensus_threshold = 0.6  # Lowered from 75% to 60% consensus required for veto
            
            # Handle opposing consensus
            if consensus_action and consensus_action != final_action:
                if consensus_level >= ai_consensus_threshold:
                    logger.info(f"Multi-AI consensus vetoed {final_action} signal for {symbol}: {consensus_action} consensus at {consensus_level:.1%}")
                    return  # Veto signal
                else:
                    logger.info(f"Weak multi-AI opposition for {symbol}: penalizing confidence (consensus: {consensus_level:.1%})")
                    # Penalize confidence for weak opposition
                    base_signal['confidence'] *= 0.6
            
            # Apply confidence adjustments from multi-AI analysis
            if 'final_confidence' in consensus:
                confidence = consensus['final_confidence']
            else:
                confidence = base_signal['confidence'] + consensus.get('confidence_adjustment', 0)
                confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]
            
            # Apply additional Manus AI confidence adjustments if available
            if consensus_recommendations and confidence is not None:
                original_confidence = confidence
                confidence = self._apply_manus_ai_confidence_adjustment(
                    confidence, strategy_name, consensus_recommendations
                )
                if abs(confidence - original_confidence) > 0.01:
                    logger.info(f"Manus AI further adjusted confidence for {symbol} {strategy_name}: "
                               f"{original_confidence:.1%} -> {confidence:.1%}")
            
            # Create signal object with enhanced data
            expires_at = datetime.utcnow() + timedelta(
                minutes=strategy_config.config.get('expiry_bars', 60)
            )
            
            # Convert numpy types to Python types for database compatibility
            price = float(base_signal['price']) if base_signal['price'] is not None else None
            sl = float(base_signal.get('sl')) if base_signal.get('sl') is not None else None
            tp = float(base_signal.get('tp')) if base_signal.get('tp') is not None else None
            confidence = float(confidence) if confidence is not None else None
            
            signal = Signal(
                symbol=symbol,
                action=final_action,
                price=price,
                sl=sl,
                tp=tp,
                confidence=confidence,
                strategy=strategy_name,
                expires_at=expires_at
            )
            
            # Handle risk flags from multi-AI consensus
            risk_flags = consensus.get('risk_flags', [])
            if risk_flags:
                high_severity_flags = [flag for flag in risk_flags if flag.get('severity', 'low') == 'high']
                if high_severity_flags:
                    signal.blocked_by_risk = True  # type: ignore[assignment]
                    signal.risk_reason = f"Multi-AI risk flags: {', '.join([f['type'] for f in high_severity_flags])}"  # type: ignore[assignment]
                    logger.info(f"Signal blocked by multi-AI risk flags: {signal.risk_reason}")
                else:
                    # Apply confidence penalty for lower severity risk flags
                    signal.confidence = max(0.0, signal.confidence - 0.1)  # type: ignore[assignment]
                    logger.info(f"Multi-AI risk flags applied confidence penalty: {[f['type'] for f in risk_flags]}")
            
            # Apply standard risk management
            if not signal.blocked_by_risk:  # type: ignore[truthy-bool]
                risk_manager = RiskManager(db)
                risk_check = risk_manager.check_signal(signal, data)
                
                if not risk_check['allowed']:
                    signal.blocked_by_risk = True  # type: ignore[assignment]
                    signal.risk_reason = risk_check['reason']  # type: ignore[assignment]
                    logger.info(f"Signal blocked by risk management: {risk_check['reason']}")
                else:
                    # Automatic trade execution if enabled and confidence threshold met
                    if self.auto_trade_enabled and signal.confidence >= self.confidence_threshold:  # type: ignore[operator]
                        try:
                            await self._execute_auto_trade(signal, db)
                        except Exception as e:
                            logger.error(f"Auto-trade execution failed for signal {signal.id}: {e}")
            
            # Save signal to database
            db.add(signal)
            db.commit()
            
            # Log comprehensive multi-AI analysis results with proper validation
            participating_agents = consensus.get('participating_agents', 0)
            consensus_strength = consensus.get('consensus_level', 0.0)
            
            # **FINAL VALIDATION**: Never log or create signals with 0 agents and high confidence
            if participating_agents == 0 and signal.confidence > 0.5:
                logger.error(f"CRITICAL BUG DETECTED: Attempted to create signal with 0 agents and {signal.confidence:.1%} confidence for {symbol} - BLOCKING")
                db.rollback()
                return
                
            logger.info(f"Multi-AI enhanced signal created for {symbol}: {final_action} @ {price} "
                       f"(confidence: {signal.confidence:.1%}, consensus: {consensus_strength:.1%}, "
                       f"agents: {participating_agents})")
            
        except Exception as e:
            strategy_name = getattr(strategy_config, 'name', 'unknown') if 'strategy_config' in locals() else 'unknown'
            logger.error(f"Error processing strategy {strategy_name} for {symbol} with multi-AI: {e}")
            db.rollback()
    
    async def _process_strategy_with_ai_consensus(
        self, 
        symbol: str, 
        data: pd.DataFrame, 
        strategy_config: Strategy, 
        db: Session,
        consensus_recommendations: Optional[Dict] = None
    ):
        """Fallback method for legacy AI consensus - delegates to standard processing"""
        logger.info(f"Using fallback AI consensus processing for {symbol}")
        await self._process_strategy(symbol, data, strategy_config, db, consensus_recommendations)
    
    async def _execute_auto_trade(self, signal: Signal, db: Session):
        """Execute automatic trade for high-confidence signals"""
        try:
            logger.info(f"Executing auto-trade for signal {signal.id}: {signal.action} {signal.symbol} @ {signal.price} (confidence: {signal.confidence:.1%})")  # type: ignore[misc]
            
            # Determine order type
            action = str(signal.action)  # Force string conversion from SQLAlchemy column
            if action == 'BUY':
                order_type = OrderType.MARKET_BUY
            elif action == 'SELL':
                order_type = OrderType.MARKET_SELL
            else:
                raise ValueError(f"Unknown signal action: {action}")
            
            # Create order request
            order_request = OrderRequest(
                symbol=signal.symbol,  # type: ignore[arg-type]
                order_type=order_type,
                volume=self.default_lot_size,
                price=None,  # Market order
                stop_loss=signal.sl,  # type: ignore[arg-type]
                take_profit=signal.tp,  # type: ignore[arg-type]
                comment=f"AutoTrade-{signal.id}-{signal.strategy}",
                magic_number=234000 + signal.id  # type: ignore[operator]  # Unique magic number
            )
            
            # Execute order through MT5 bridge
            execution_result = await self.execution_provider.execute_order(order_request)
            
            if execution_result.success:
                # Update signal with execution details
                signal.auto_traded = True  # type: ignore[assignment]
                signal.broker_ticket = execution_result.ticket  # type: ignore[assignment]
                signal.executed_price = execution_result.executed_price  # type: ignore[assignment]
                signal.executed_volume = execution_result.executed_volume  # type: ignore[assignment]
                signal.execution_slippage = execution_result.slippage  # type: ignore[assignment]
                signal.execution_time = execution_result.execution_time  # type: ignore[assignment]
                
                logger.info(f"Auto-trade executed successfully - Ticket: {execution_result.ticket}, Price: {execution_result.executed_price}")
                
            else:
                # Log execution failure
                signal.auto_trade_failed = True  # type: ignore[assignment]
                signal.execution_error = execution_result.message  # type: ignore[assignment]
                
                logger.error(f"Auto-trade execution failed: {execution_result.message}")
                
        except Exception as e:
            # Update signal with failure info
            signal.auto_trade_failed = True  # type: ignore[assignment]
            signal.execution_error = str(e)  # type: ignore[assignment]
            
            logger.error(f"Auto-trade execution error: {e}")
            raise
    
    def _is_duplicate_signal(self, symbol: str, signal_data: dict, strategy_name: str, db: Session) -> bool:
        """Check if this signal conflicts with recent signals (enhanced logic)"""
        try:
            now = datetime.utcnow()
            cooldown_minutes = 15  # No new signals for same symbol within 15 minutes
            
            # Check for any recent signal for this symbol (regardless of strategy)
            recent_signal = db.query(Signal).filter(
                Signal.symbol == symbol,
                Signal.issued_at > now - timedelta(minutes=cooldown_minutes)
            ).order_by(Signal.issued_at.desc()).first()
            
            if recent_signal:
                # If recent signal has same action, it's a duplicate
                if str(recent_signal.action) == signal_data['action']:  # type: ignore[operator]
                    logger.debug(f"Duplicate signal blocked for {symbol}: same action within cooldown")
                    return True
                    
                # If recent signal has opposite action, check confidence levels
                if str(recent_signal.action) != signal_data['action']:  # type: ignore[operator]
                    # Only allow opposite signal if new signal has significantly higher confidence
                    confidence_threshold = float(recent_signal.confidence) + 0.15  # type: ignore[operator]  # 15% higher confidence required
                    if signal_data['confidence'] < confidence_threshold:
                        logger.debug(f"Conflicting signal blocked for {symbol}: insufficient confidence {signal_data['confidence']:.2f} vs required {confidence_threshold:.2f}")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking duplicate signal: {e}")
            return False
            
    def _get_cross_strategy_consensus(self, symbol: str, signal_data: dict, db: Session) -> bool:
        """Check if multiple strategies agree on signal direction"""
        try:
            now = datetime.utcnow()
            recent_minutes = 10  # Look at signals in last 10 minutes
            
            # Get recent signals for this symbol from all strategies
            recent_signals = db.query(Signal).filter(
                Signal.symbol == symbol,
                Signal.issued_at > now - timedelta(minutes=recent_minutes),
                Signal.blocked_by_risk.is_(False)  # type: ignore[attr-defined]
            ).all()
            
            if not recent_signals:
                return True  # No recent signals, allow
                
            # Count signals by action
            buy_count = sum(1 for s in recent_signals if str(s.action) == 'BUY')
            sell_count = sum(1 for s in recent_signals if str(s.action) == 'SELL')
            
            # Current signal action
            current_action = signal_data['action']
            
            # If there are conflicting signals, require higher confidence (adaptive threshold)
            if (current_action == 'BUY' and sell_count > 0) or (current_action == 'SELL' and buy_count > 0):
                # **FIXED ADAPTIVE THRESHOLD**: Use same 2-agent threshold as consensus system  
                # 50% for 2 agents, 65% for 3 agents, 80% for 4+ agents
                participating_agents = signal_data.get('participating_agents', 3)
                min_confidence = 0.50 if participating_agents == 2 else (0.65 if participating_agents <= 3 else 0.80)
                if signal_data['confidence'] < min_confidence:
                    logger.debug(f"Cross-strategy conflict: {symbol} needs {min_confidence*100:.0f}%+ confidence for {current_action}, got {signal_data['confidence']:.2f}")
                    return False
                else:
                    logger.info(f"Cross-strategy conflict resolved: {symbol} {current_action} approved with {signal_data['confidence']:.1%} confidence (threshold: {min_confidence:.1%})")
                    
            return True
            
        except Exception as e:
            logger.error(f"Error checking cross-strategy consensus: {e}")
            return True  # Allow on error
    
    async def _get_manus_ai_recommendations(self, symbol: str, market_data: pd.DataFrame) -> Optional[Dict]:
        """Get professional strategy recommendations from enhanced Manus AI"""
        try:
            # Use Manus AI to get intelligent strategy recommendations
            recommendations = self.manus_ai.suggest_strategies(symbol, market_data)
            
            if recommendations.get('status') == 'success':
                logger.info(f"Manus AI recommendations for {symbol}: "
                           f"regime={recommendations['market_analysis']['regime']}, "
                           f"top_strategies={[s['name'] for s in recommendations['recommended_strategies'][:3]]}")
                return recommendations
            else:
                logger.debug(f"Manus AI recommendations unavailable for {symbol}, using fallback")
                return None
                
        except Exception as e:
            logger.warning(f"Error getting Manus AI recommendations for {symbol}: {e}")
            return None
    
    async def _get_multi_ai_recommendations(self, symbol: str, data: pd.DataFrame, strategy_names: List[str]) -> Dict[str, Any]:
        """Get enhanced recommendations from Multi-AI Consensus system"""
        try:
            # Check if multi-AI consensus is available
            if self.multi_ai_consensus is None:
                logger.warning(f"Multi-AI consensus not available for {symbol}, using fallback")
                raise Exception("Multi-AI consensus system not initialized")
                
            # Generate comprehensive multi-AI analysis
            multi_ai_analysis = await self.multi_ai_consensus.generate_enhanced_signal_analysis(
                symbol=symbol,
                market_data=data,
                base_signal=None  # Will be provided later during signal processing
            )
            
            # Extract core recommendations from consensus
            manus_insights = multi_ai_analysis.get('agent_insights', {}).get('manus_ai', {})
            
            return {
                'regime': manus_insights.get('regime', 'UNKNOWN'),
                'confidence': multi_ai_analysis.get('final_confidence', 0.5),
                'recommended_strategies': strategy_names,  # All strategies considered with AI adjustment
                'market_condition': manus_insights.get('market_condition', 'unknown'),
                'ai_mode': 'multi_ai_consensus',
                'agent_count': multi_ai_analysis.get('agent_count', 1),
                'consensus_strength': multi_ai_analysis.get('consensus_strength', 0.5),
                'multi_ai_analysis': multi_ai_analysis  # Store full analysis for later use
            }
        except Exception as e:
            logger.error(f"Multi-AI recommendations failed for {symbol}: {e}")
            # Fallback to Manus AI only
            fallback = await self._get_manus_ai_recommendations(symbol, data)
            if fallback:
                return {
                    'regime': fallback.get('market_analysis', {}).get('regime', 'UNKNOWN'),
                    'confidence': fallback.get('market_analysis', {}).get('regime_confidence', 0.5),
                    'recommended_strategies': [s['name'] for s in fallback.get('recommended_strategies', [])],
                    'market_condition': 'unknown',
                    'ai_mode': 'manus_fallback'
                }
            else:
                return {
                    'regime': 'UNKNOWN',
                    'confidence': 0.5,
                    'recommended_strategies': strategy_names,
                    'market_condition': 'unknown',
                    'ai_mode': 'fallback'
                }
    
    def _should_block_strategy(self, strategy_name: str, manus_recommendations: Optional[Dict]) -> Tuple[bool, str]:
        """
        CRITICAL: Check if strategy should be BLOCKED based on Manus AI "avoid" recommendations
        
        This method enforces strategy guardrails by actively blocking strategies that Manus AI
        recommends to avoid for the current market conditions.
        
        Returns:
            Tuple of (should_block, reason)
        """
        try:
            if not manus_recommendations or manus_recommendations.get('status') != 'success':
                return False, "No Manus AI recommendations available - allowing strategy"
            
            market_analysis = manus_recommendations.get('market_analysis', {})
            regime = market_analysis.get('regime', 'UNKNOWN')
            regime_confidence = market_analysis.get('regime_confidence', 0.0)
            
            # Define confidence threshold for enforcing blocks
            # Only block when we're confident about the market regime
            confidence_threshold = 0.65  # 65% confidence threshold
            
            if regime_confidence < confidence_threshold:
                return False, f"Regime confidence too low ({regime_confidence:.1%}) - allowing strategy"
            
            # Get the regime-specific strategy mapping from Manus AI
            strategy_mapping = {
                'TRENDING': {
                    'avoid': ['meanrev_bb', 'stochastic'],
                    'reasoning': 'Trending markets favor breakout and momentum strategies'
                },
                'STRONG_TRENDING': {
                    'avoid': ['meanrev_bb', 'stochastic', 'rsi_divergence'],
                    'reasoning': 'Strong trends require momentum strategies with wider stops'
                },
                'RANGING': {
                    'avoid': ['donchian_atr', 'fibonacci'],
                    'reasoning': 'Range-bound markets favor mean reversion strategies'
                },
                'HIGH_VOLATILITY': {
                    'avoid': ['donchian_atr'],
                    'reasoning': 'High volatility requires precision timing strategies'
                }
            }
            
            # Check if current strategy should be blocked for this regime
            if regime in strategy_mapping:
                avoid_strategies = strategy_mapping[regime]['avoid']
                reasoning = strategy_mapping[regime]['reasoning']
                
                if strategy_name in avoid_strategies:
                    return True, f"Manus AI blocks {strategy_name} for {regime} regime (confidence: {regime_confidence:.1%}) - {reasoning}"
            
            # Additional check: look for explicit avoid recommendations in the response
            recommended_strategies = manus_recommendations.get('recommended_strategies', [])
            for rec in recommended_strategies:
                if (rec['name'] == strategy_name and 
                    rec.get('recommended') == False and 
                    rec.get('confidence', 0) < 0.3):  # Very low confidence = avoid
                    return True, f"Manus AI explicitly recommends avoiding {strategy_name} (confidence: {rec.get('confidence', 0):.1%})"
            
            return False, f"Strategy {strategy_name} allowed for {regime} regime"
            
        except Exception as e:
            logger.error(f"Error checking strategy blocking: {e}")
            return False, "Error in strategy blocking check - allowing strategy as fallback"
    
    def _should_prioritize_strategy(self, strategy_name: str, manus_recommendations: Optional[Dict]) -> Tuple[bool, str]:
        """
        Check if strategy should be prioritized based on Manus AI recommendations
        
        Returns:
            Tuple of (should_prioritize, reason)
        """
        try:
            if not manus_recommendations or manus_recommendations.get('status') != 'success':
                return False, "No Manus AI recommendations available"
            
            recommended_strategies = manus_recommendations.get('recommended_strategies', [])
            
            # Check if this strategy is in the top recommendations
            for rec in recommended_strategies[:3]:  # Top 3 strategies
                if rec['name'] == strategy_name and rec.get('recommended', False):
                    confidence = rec.get('confidence', 0)
                    priority = rec.get('priority', 'tertiary')
                    
                    if priority == 'primary' and confidence >= 0.7:
                        return True, f"Manus AI primary recommendation (confidence: {confidence:.1%})"
                    elif priority == 'secondary' and confidence >= 0.6:
                        return True, f"Manus AI secondary recommendation (confidence: {confidence:.1%})"
            
            return False, "Strategy not in top Manus AI recommendations"
        
        except Exception as e:
            logger.error(f"Error checking strategy prioritization: {e}")
            return False, "Error in strategy prioritization"
    
    def _apply_manus_ai_confidence_adjustment(
        self, 
        original_confidence: float, 
        strategy_name: str, 
        manus_recommendations: Dict
    ) -> float:
        """
        Apply Manus AI-based confidence adjustments to signal confidence
        
        Args:
            original_confidence: Original strategy confidence (0.0 to 1.0)
            strategy_name: Name of the strategy generating the signal
            manus_recommendations: Manus AI recommendations dict
            
        Returns:
            Adjusted confidence (0.0 to 1.0)
        """
        try:
            if manus_recommendations.get('status') != 'success':
                return original_confidence
            
            adjusted_confidence = original_confidence
            recommended_strategies = manus_recommendations.get('recommended_strategies', [])
            
            # Find the strategy in recommendations
            strategy_rec = None
            for rec in recommended_strategies:
                if rec['name'] == strategy_name:
                    strategy_rec = rec
                    break
            
            if not strategy_rec:
                # Strategy not in recommendations - small confidence reduction
                adjusted_confidence *= 0.95
                logger.debug(f"Strategy {strategy_name} not in Manus AI recommendations - small confidence reduction")
                return min(max(adjusted_confidence, 0.1), 1.0)
            
            # Apply adjustments based on Manus AI analysis
            manus_confidence = strategy_rec.get('confidence', 0.5)
            priority = strategy_rec.get('priority', 'tertiary')
            recommended = strategy_rec.get('recommended', False)
            
            # Priority-based adjustments
            if priority == 'primary' and recommended:
                adjustment_factor = 1.1  # 10% boost for primary strategies
            elif priority == 'secondary' and recommended:
                adjustment_factor = 1.05  # 5% boost for secondary strategies
            elif not recommended:
                adjustment_factor = 0.85  # 15% reduction for non-recommended strategies
            else:
                adjustment_factor = 1.0  # No adjustment for tertiary recommended
            
            # Confidence alignment adjustment
            confidence_diff = abs(manus_confidence - original_confidence)
            if confidence_diff > 0.2:  # Large difference
                # Move original confidence towards Manus AI confidence
                blend_factor = 0.3  # 30% towards Manus AI assessment
                adjusted_confidence = original_confidence * (1 - blend_factor) + manus_confidence * blend_factor
            
            # Apply the priority adjustment
            adjusted_confidence *= adjustment_factor
            
            # Market condition adjustments from Manus AI analysis
            market_analysis = manus_recommendations.get('market_analysis', {})
            volatility_level = market_analysis.get('volatility_level', 'medium')
            
            if volatility_level == 'high':
                adjusted_confidence *= 0.9  # Reduce confidence in high volatility
                logger.debug(f"High volatility detected - reducing confidence for {strategy_name}")
            
            # Ensure confidence stays within bounds
            adjusted_confidence = min(max(adjusted_confidence, 0.1), 1.0)
            
            return adjusted_confidence
            
        except Exception as e:
            logger.error(f"Error applying Manus AI confidence adjustment: {e}")
            return original_confidence
    
    # ========== NEW DUAL-AI CONSENSUS METHODS ==========
    
    def _should_block_strategy_ai_consensus(self, strategy_name: str, consensus_recommendations: Optional[Dict]) -> Tuple[bool, str]:
        """
        Check if strategy should be BLOCKED based on AI consensus recommendations
        Enhanced version that considers both Manus AI and ChatGPT opinions
        """
        try:
            if not consensus_recommendations:
                return False, "No AI consensus recommendations available - allowing strategy"
            
            # Check if we have consensus result
            if 'consensus_result' in consensus_recommendations:
                consensus_result = consensus_recommendations['consensus_result']
                
                # Check AI consensus confidence threshold
                if hasattr(consensus_result, 'overall_confidence'):
                    if consensus_result.overall_confidence < self.ai_consensus_threshold:
                        return True, f"AI consensus confidence too low ({consensus_result.overall_confidence:.1%}) - blocking strategy"
                
                # Check if strategy is in avoid list from either AI
                if hasattr(consensus_result, 'ai_contributions'):
                    ai_contributions = consensus_result.ai_contributions
                    
                    # Check Manus AI contribution
                    manus_contrib = ai_contributions.get('manus_ai', {})
                    if 'avoid_strategies' in manus_contrib:
                        if strategy_name in manus_contrib['avoid_strategies']:
                            return True, f"Manus AI (via consensus) recommends avoiding {strategy_name}"
                    
                    # Check ChatGPT contribution  
                    chatgpt_contrib = ai_contributions.get('chatgpt', {})
                    if 'avoid_strategies' in chatgpt_contrib:
                        if strategy_name in chatgpt_contrib['avoid_strategies']:
                            return True, f"ChatGPT (via consensus) recommends avoiding {strategy_name}"
                
                # Check conflict areas
                if hasattr(consensus_result, 'conflict_areas') and 'strategy_selection' in consensus_result.conflict_areas:
                    # In case of strategy selection conflicts, be more conservative
                    recommended_strategies = consensus_result.recommended_strategies if hasattr(consensus_result, 'recommended_strategies') else []
                    strategy_names = [s.get('name', '') for s in recommended_strategies]
                    
                    if strategy_name not in strategy_names:
                        return True, f"Strategy {strategy_name} not recommended due to AI disagreement"
                
                return False, f"AI consensus allows strategy {strategy_name}"
            
            else:
                # Fallback to single Manus AI logic
                return self._should_block_strategy(strategy_name, consensus_recommendations)
                
        except Exception as e:
            logger.error(f"Error checking AI consensus strategy blocking: {e}")
            return False, "Error in AI consensus strategy blocking - allowing as fallback"
    
    def _should_prioritize_strategy_ai_consensus(self, strategy_name: str, consensus_recommendations: Optional[Dict]) -> Tuple[bool, str]:
        """
        Check if strategy should be prioritized based on AI consensus
        Enhanced version considering both AIs
        """
        try:
            if not consensus_recommendations:
                return False, "No AI consensus recommendations available"
            
            # Check if we have consensus result
            if 'consensus_result' in consensus_recommendations:
                consensus_result = consensus_recommendations['consensus_result']
                
                if hasattr(consensus_result, 'recommended_strategies'):
                    recommended_strategies = consensus_result.recommended_strategies
                    
                    # Check if strategy is in top recommendations
                    for i, strategy_rec in enumerate(recommended_strategies[:3]):  # Top 3
                        if strategy_rec.get('name') == strategy_name:
                            confidence = strategy_rec.get('confidence', 0.5)
                            consensus_conf = getattr(consensus_result, 'overall_confidence', 0.5)
                            
                            # High priority if in top recommendation with good consensus
                            if i == 0 and confidence >= 0.7 and consensus_conf >= 0.7:
                                consensus_level = getattr(consensus_result, 'consensus_level', 'unknown')
                                return True, f"AI consensus top recommendation (confidence: {confidence:.1%}, consensus: {consensus_level})"
                            
                            # Medium priority for other top strategies
                            elif i < 3 and confidence >= 0.6:
                                return True, f"AI consensus recommendation #{i+1} (confidence: {confidence:.1%})"
                
                return False, "Strategy not in top AI consensus recommendations"
            
            else:
                # Fallback to single Manus AI logic
                return self._should_prioritize_strategy(strategy_name, consensus_recommendations)
                
        except Exception as e:
            logger.error(f"Error checking AI consensus strategy prioritization: {e}")
            return False, "Error in AI consensus prioritization"
    
    async def _enhance_signal_with_ai_consensus(
        self, 
        signal_data: Dict, 
        strategy_name: str, 
        symbol: str, 
        market_data: pd.DataFrame,
        consensus_recommendations: Dict
    ) -> Optional[Dict]:
        """
        Enhance signal using AI consensus validation and refinement
        """
        try:
            if not consensus_recommendations or 'consensus_result' not in consensus_recommendations:
                return signal_data
            
            consensus_result = consensus_recommendations['consensus_result']
            
            # Validate signal against AI consensus
            if not await self._validate_signal_with_ai_consensus(signal_data, strategy_name, consensus_result):
                logger.info(f"Signal for {symbol} {strategy_name} rejected by AI consensus validation")
                return None
            
            # Enhance signal with AI confidence and metrics
            enhanced_signal = signal_data.copy()
            
            # Add AI consensus metadata
            enhanced_signal['ai_consensus_confidence'] = getattr(consensus_result, 'overall_confidence', 0.5)
            enhanced_signal['consensus_level'] = getattr(consensus_result, 'consensus_level', 'unknown')
            enhanced_signal['ai_reasoning'] = getattr(consensus_result, 'reasoning', '')
            
            # Adjust confidence based on AI consensus
            original_confidence = signal_data.get('confidence', 0.5)
            ai_confidence_boost = self._calculate_ai_confidence_boost(strategy_name, consensus_result)
            enhanced_confidence = min(1.0, original_confidence + ai_confidence_boost)
            enhanced_signal['confidence'] = enhanced_confidence
            
            # Add AI-enhanced stop loss and take profit if available
            ai_risk_params = await self._get_ai_enhanced_risk_parameters(
                signal_data, symbol, market_data, consensus_result
            )
            if ai_risk_params:
                enhanced_signal.update(ai_risk_params)
            
            logger.debug(f"Signal enhanced by AI consensus for {symbol} {strategy_name}: "
                        f"confidence {original_confidence:.2f} -> {enhanced_confidence:.2f}")
            
            return enhanced_signal
            
        except Exception as e:
            logger.error(f"Error enhancing signal with AI consensus: {e}")
            return signal_data  # Return original signal on error
    
    async def _validate_signal_with_ai_consensus(
        self, 
        signal_data: Dict, 
        strategy_name: str, 
        consensus_result
    ) -> bool:
        """
        Validate signal against AI consensus recommendations
        """
        try:
            # Check if strategy is recommended by consensus
            if hasattr(consensus_result, 'recommended_strategies'):
                recommended_strategies = [s.get('name', '') for s in consensus_result.recommended_strategies]
                if strategy_name not in recommended_strategies:
                    return False
            
            # Check minimum confidence threshold
            signal_confidence = signal_data.get('confidence', 0.5)
            consensus_confidence = getattr(consensus_result, 'overall_confidence', 0.5)
            
            # Require minimum combined confidence
            combined_confidence = (signal_confidence + consensus_confidence) / 2
            if combined_confidence < 0.6:  # 60% minimum combined confidence
                return False
            
            # Check for high disagreement between AIs
            if hasattr(consensus_result, 'consensus_level'):
                from ..services.ai_strategy_consensus import ConsensusLevel
                if consensus_result.consensus_level == ConsensusLevel.DISAGREEMENT:
                    # Require very high signal confidence to override AI disagreement
                    if signal_confidence < 0.85:
                        return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error validating signal with AI consensus: {e}")
            return True  # Allow signal on validation error
    
    def _calculate_ai_confidence_boost(self, strategy_name: str, consensus_result) -> float:
        """
        Calculate confidence boost based on AI consensus
        """
        try:
            boost = 0.0
            
            if hasattr(consensus_result, 'recommended_strategies'):
                for strategy_rec in consensus_result.recommended_strategies:
                    if strategy_rec.get('name') == strategy_name:
                        # Boost based on consensus confidence and strategy ranking
                        consensus_conf = getattr(consensus_result, 'overall_confidence', 0.5)
                        strategy_conf = strategy_rec.get('confidence', 0.5)
                        
                        # Higher boost for higher consensus and strategy confidence
                        boost = min(0.15, (consensus_conf - 0.5) * 0.3 + (strategy_conf - 0.5) * 0.2)
                        break
            
            # Additional boost for high agreement
            if hasattr(consensus_result, 'agreement_score'):
                agreement_score = consensus_result.agreement_score
                if agreement_score > 0.8:
                    boost += 0.05  # Extra 5% for high agreement
            
            return max(0.0, boost)
            
        except Exception as e:
            logger.warning(f"Error calculating AI confidence boost: {e}")
            return 0.0
    
    async def _get_ai_enhanced_risk_parameters(
        self, 
        signal_data: Dict, 
        symbol: str, 
        market_data: pd.DataFrame,
        consensus_result
    ) -> Optional[Dict]:
        """
        Get AI-enhanced risk parameters for stop loss and take profit
        """
        try:
            risk_params = {}
            
            # Get AI risk guidance from consensus
            if hasattr(consensus_result, 'ai_contributions'):
                ai_contributions = consensus_result.ai_contributions
                
                # Check if any AI provided specific risk parameters
                for ai_name, contribution in ai_contributions.items():
                    if 'risk_parameters' in contribution:
                        ai_risk = contribution['risk_parameters']
                        
                        # Use AI-suggested stop loss if available and reasonable
                        if 'suggested_sl_pct' in ai_risk:
                            sl_pct = ai_risk['suggested_sl_pct']
                            if 0.005 <= sl_pct <= 0.05:  # 0.5% to 5% reasonable range
                                current_price = signal_data.get('price', 0)
                                if current_price > 0:
                                    if signal_data.get('action') == 'BUY':
                                        risk_params['sl'] = current_price * (1 - sl_pct)
                                    else:
                                        risk_params['sl'] = current_price * (1 + sl_pct)
                        
                        # Use AI-suggested take profit if available
                        if 'suggested_tp_pct' in ai_risk:
                            tp_pct = ai_risk['suggested_tp_pct']
                            if 0.01 <= tp_pct <= 0.10:  # 1% to 10% reasonable range
                                current_price = signal_data.get('price', 0)
                                if current_price > 0:
                                    if signal_data.get('action') == 'BUY':
                                        risk_params['tp'] = current_price * (1 + tp_pct)
                                    else:
                                        risk_params['tp'] = current_price * (1 - tp_pct)
            
            return risk_params if risk_params else None
            
        except Exception as e:
            logger.warning(f"Error getting AI-enhanced risk parameters: {e}")
            return None
    
    def _apply_ai_consensus_confidence_adjustment(
        self, 
        original_confidence: float, 
        strategy_name: str, 
        consensus_recommendations: Dict
    ) -> float:
        """
        Apply AI consensus-based confidence adjustments
        """
        try:
            if 'consensus_result' not in consensus_recommendations:
                return self._apply_manus_ai_confidence_adjustment(original_confidence, strategy_name, consensus_recommendations)
            
            consensus_result = consensus_recommendations['consensus_result']
            adjusted_confidence = original_confidence
            
            # Boost confidence based on consensus level
            if hasattr(consensus_result, 'consensus_level'):
                from ..services.ai_strategy_consensus import ConsensusLevel
                
                if consensus_result.consensus_level == ConsensusLevel.HIGH_AGREEMENT:
                    adjusted_confidence *= 1.10  # 10% boost for high agreement
                elif consensus_result.consensus_level == ConsensusLevel.MODERATE_AGREEMENT:
                    adjusted_confidence *= 1.05  # 5% boost for moderate agreement
                elif consensus_result.consensus_level == ConsensusLevel.DISAGREEMENT:
                    adjusted_confidence *= 0.85  # 15% reduction for disagreement
            
            # Adjust based on overall consensus confidence
            if hasattr(consensus_result, 'overall_confidence'):
                consensus_conf = consensus_result.overall_confidence
                confidence_factor = 0.8 + (consensus_conf * 0.4)  # Factor between 0.8 and 1.2
                adjusted_confidence *= confidence_factor
            
            # Find strategy-specific adjustments
            if hasattr(consensus_result, 'recommended_strategies'):
                for strategy_rec in consensus_result.recommended_strategies:
                    if strategy_rec.get('name') == strategy_name:
                        strategy_conf = strategy_rec.get('confidence', 0.5)
                        # Additional boost if strategy has high individual confidence
                        if strategy_conf > 0.8:
                            adjusted_confidence *= 1.05
                        elif strategy_conf < 0.4:
                            adjusted_confidence *= 0.90
                        break
            
            # Ensure confidence stays within bounds
            return min(max(adjusted_confidence, 0.1), 1.0)
            
        except Exception as e:
            logger.error(f"Error applying AI consensus confidence adjustment: {e}")
            return original_confidence
