"""
Comprehensive Instrument Metadata Database
Provides accurate specifications for forex, crypto, and metals trading
"""
from typing import Dict, Any, Optional, Literal
from dataclasses import dataclass
from enum import Enum

class AssetClass(Enum):
    """Asset class classification"""
    FOREX = "forex"
    CRYPTO = "crypto"
    METALS = "metals"
    OIL = "oil"
    INDICES = "indices"

@dataclass
class InstrumentMetadata:
    """Complete instrument specification"""
    symbol: str
    asset_class: AssetClass
    
    # Price specifications
    pip_size: float                # Minimum price movement (pip/tick size)
    decimal_places: int           # Number of decimal places for display
    quote_precision: int          # Quote price precision
    
    # Order specifications  
    min_lot_size: float           # Minimum order size
    max_lot_size: float           # Maximum order size
    lot_step: float               # Lot size increment
    
    # Value calculations
    pip_value_per_lot: float      # Value of 1 pip for 1 lot in USD
    contract_size: int            # Contract size (e.g., 100,000 for forex)
    
    # Market hours (in UTC)
    market_open_days: list        # List of weekdays when market is open (0=Monday)
    market_open_hours: tuple      # (start_hour, end_hour) in UTC
    is_24_7: bool                 # True for crypto markets
    
    # Additional specifications
    margin_percentage: float      # Margin requirement percentage
    description: str              # Human-readable description
    base_currency: str           # Base currency
    quote_currency: str          # Quote currency

class InstrumentMetadataDB:
    """Comprehensive instrument metadata database"""
    
    def __init__(self):
        self._instruments = self._build_metadata_db()
    
    def _build_metadata_db(self) -> Dict[str, InstrumentMetadata]:
        """Build the complete instrument metadata database"""
        instruments = {}
        
        # === ALL MAJOR FOREX PAIRS CONFIGURATION ===
        # Complete set of major forex pairs for comprehensive trading coverage
        # Standard institutional forex trading specifications with modern 5/3 decimal precision
        
        # === MAJOR USD FOREX PAIRS ===
        # Using modern forex specifications (5 decimal for standard pairs, 3 for JPY pairs)
        forex_pairs = {
            # USD Major Pairs
            'EURUSD': InstrumentMetadata(
                symbol='EURUSD', asset_class=AssetClass.FOREX,
                pip_size=0.00001, decimal_places=5, quote_precision=5,  # Modern 5-decimal EUR/USD
                min_lot_size=0.01, max_lot_size=1000.0, lot_step=0.01,
                pip_value_per_lot=1.0, contract_size=100000,  # $1 per pip per standard lot
                market_open_days=[0,1,2,3,4], market_open_hours=(0,24), is_24_7=True,  # Sun 21:00 - Fri 21:00 UTC (24hr weekdays)
                margin_percentage=0.02, description="Euro vs US Dollar",
                base_currency='EUR', quote_currency='USD'
            ),
            'GBPUSD': InstrumentMetadata(
                symbol='GBPUSD', asset_class=AssetClass.FOREX,
                pip_size=0.00001, decimal_places=5, quote_precision=5,  # Modern 5-decimal GBP/USD
                min_lot_size=0.01, max_lot_size=1000.0, lot_step=0.01,
                pip_value_per_lot=1.0, contract_size=100000,  # $1 per pip per standard lot
                market_open_days=[0,1,2,3,4], market_open_hours=(0,24), is_24_7=True,  # Sun 21:00 - Fri 21:00 UTC (24hr weekdays)
                margin_percentage=0.02, description="British Pound vs US Dollar",
                base_currency='GBP', quote_currency='USD'
            ),
            'USDJPY': InstrumentMetadata(
                symbol='USDJPY', asset_class=AssetClass.FOREX,
                pip_size=0.001, decimal_places=3, quote_precision=3,  # Modern 3-decimal USD/JPY
                min_lot_size=0.01, max_lot_size=1000.0, lot_step=0.01,
                pip_value_per_lot=0.67, contract_size=100000,  # ~$0.67 per pip per standard lot (varies with JPY rate)
                market_open_days=[0,1,2,3,4], market_open_hours=(0,24), is_24_7=True,  # Sun 21:00 - Fri 21:00 UTC (24hr weekdays)
                margin_percentage=0.02, description="US Dollar vs Japanese Yen",
                base_currency='USD', quote_currency='JPY'
            ),
            'USDCHF': InstrumentMetadata(
                symbol='USDCHF', asset_class=AssetClass.FOREX,
                pip_size=0.00001, decimal_places=5, quote_precision=5,  # Modern 5-decimal USD/CHF
                min_lot_size=0.01, max_lot_size=1000.0, lot_step=0.01,
                pip_value_per_lot=1.08, contract_size=100000,  # ~$1.08 per pip per standard lot
                market_open_days=[0,1,2,3,4], market_open_hours=(0,24), is_24_7=True,
                margin_percentage=0.02, description="US Dollar vs Swiss Franc",
                base_currency='USD', quote_currency='CHF'
            ),
            'AUDUSD': InstrumentMetadata(
                symbol='AUDUSD', asset_class=AssetClass.FOREX,
                pip_size=0.00001, decimal_places=5, quote_precision=5,  # Modern 5-decimal AUD/USD
                min_lot_size=0.01, max_lot_size=1000.0, lot_step=0.01,
                pip_value_per_lot=1.0, contract_size=100000,  # $1 per pip per standard lot
                market_open_days=[0,1,2,3,4], market_open_hours=(0,24), is_24_7=True,
                margin_percentage=0.03, description="Australian Dollar vs US Dollar",
                base_currency='AUD', quote_currency='USD'
            ),
            'USDCAD': InstrumentMetadata(
                symbol='USDCAD', asset_class=AssetClass.FOREX,
                pip_size=0.00001, decimal_places=5, quote_precision=5,  # Modern 5-decimal USD/CAD
                min_lot_size=0.01, max_lot_size=1000.0, lot_step=0.01,
                pip_value_per_lot=0.74, contract_size=100000,  # ~$0.74 per pip per standard lot
                market_open_days=[0,1,2,3,4], market_open_hours=(0,24), is_24_7=True,
                margin_percentage=0.02, description="US Dollar vs Canadian Dollar",
                base_currency='USD', quote_currency='CAD'
            ),
            'NZDUSD': InstrumentMetadata(
                symbol='NZDUSD', asset_class=AssetClass.FOREX,
                pip_size=0.00001, decimal_places=5, quote_precision=5,  # Modern 5-decimal NZD/USD
                min_lot_size=0.01, max_lot_size=1000.0, lot_step=0.01,
                pip_value_per_lot=1.0, contract_size=100000,  # $1 per pip per standard lot
                market_open_days=[0,1,2,3,4], market_open_hours=(0,24), is_24_7=True,
                margin_percentage=0.03, description="New Zealand Dollar vs US Dollar",
                base_currency='NZD', quote_currency='USD'
            ),
            
            # === EUR CROSS PAIRS ===
            'EURGBP': InstrumentMetadata(
                symbol='EURGBP', asset_class=AssetClass.FOREX,
                pip_size=0.00001, decimal_places=5, quote_precision=5,  # Modern 5-decimal EUR/GBP
                min_lot_size=0.01, max_lot_size=1000.0, lot_step=0.01,
                pip_value_per_lot=1.27, contract_size=100000,  # ~$1.27 per pip per standard lot
                market_open_days=[0,1,2,3,4], market_open_hours=(0,24), is_24_7=True,
                margin_percentage=0.02, description="Euro vs British Pound",
                base_currency='EUR', quote_currency='GBP'
            ),
            'EURJPY': InstrumentMetadata(
                symbol='EURJPY', asset_class=AssetClass.FOREX,
                pip_size=0.001, decimal_places=3, quote_precision=3,  # Modern 3-decimal EUR/JPY
                min_lot_size=0.01, max_lot_size=1000.0, lot_step=0.01,
                pip_value_per_lot=0.67, contract_size=100000,  # ~$0.67 per pip per standard lot
                market_open_days=[0,1,2,3,4], market_open_hours=(0,24), is_24_7=True,
                margin_percentage=0.02, description="Euro vs Japanese Yen",
                base_currency='EUR', quote_currency='JPY'
            ),
            'EURCHF': InstrumentMetadata(
                symbol='EURCHF', asset_class=AssetClass.FOREX,
                pip_size=0.00001, decimal_places=5, quote_precision=5,  # Modern 5-decimal EUR/CHF
                min_lot_size=0.01, max_lot_size=1000.0, lot_step=0.01,
                pip_value_per_lot=1.08, contract_size=100000,  # ~$1.08 per pip per standard lot
                market_open_days=[0,1,2,3,4], market_open_hours=(0,24), is_24_7=True,
                margin_percentage=0.02, description="Euro vs Swiss Franc",
                base_currency='EUR', quote_currency='CHF'
            ),
            'EURAUD': InstrumentMetadata(
                symbol='EURAUD', asset_class=AssetClass.FOREX,
                pip_size=0.00001, decimal_places=5, quote_precision=5,  # Modern 5-decimal EUR/AUD
                min_lot_size=0.01, max_lot_size=1000.0, lot_step=0.01,
                pip_value_per_lot=0.66, contract_size=100000,  # ~$0.66 per pip per standard lot
                market_open_days=[0,1,2,3,4], market_open_hours=(0,24), is_24_7=True,
                margin_percentage=0.03, description="Euro vs Australian Dollar",
                base_currency='EUR', quote_currency='AUD'
            ),
            'EURCAD': InstrumentMetadata(
                symbol='EURCAD', asset_class=AssetClass.FOREX,
                pip_size=0.00001, decimal_places=5, quote_precision=5,  # Modern 5-decimal EUR/CAD
                min_lot_size=0.01, max_lot_size=1000.0, lot_step=0.01,
                pip_value_per_lot=0.74, contract_size=100000,  # ~$0.74 per pip per standard lot
                market_open_days=[0,1,2,3,4], market_open_hours=(0,24), is_24_7=True,
                margin_percentage=0.03, description="Euro vs Canadian Dollar",
                base_currency='EUR', quote_currency='CAD'
            ),
            
            # === GBP CROSS PAIRS ===
            'GBPJPY': InstrumentMetadata(
                symbol='GBPJPY', asset_class=AssetClass.FOREX,
                pip_size=0.001, decimal_places=3, quote_precision=3,  # Modern 3-decimal GBP/JPY
                min_lot_size=0.01, max_lot_size=1000.0, lot_step=0.01,
                pip_value_per_lot=0.67, contract_size=100000,  # ~$0.67 per pip per standard lot
                market_open_days=[0,1,2,3,4], market_open_hours=(0,24), is_24_7=True,
                margin_percentage=0.03, description="British Pound vs Japanese Yen",
                base_currency='GBP', quote_currency='JPY'
            ),
            'GBPAUD': InstrumentMetadata(
                symbol='GBPAUD', asset_class=AssetClass.FOREX,
                pip_size=0.00001, decimal_places=5, quote_precision=5,  # Modern 5-decimal GBP/AUD
                min_lot_size=0.01, max_lot_size=1000.0, lot_step=0.01,
                pip_value_per_lot=0.66, contract_size=100000,  # ~$0.66 per pip per standard lot
                market_open_days=[0,1,2,3,4], market_open_hours=(0,24), is_24_7=True,
                margin_percentage=0.03, description="British Pound vs Australian Dollar",
                base_currency='GBP', quote_currency='AUD'
            ),
            'GBPCHF': InstrumentMetadata(
                symbol='GBPCHF', asset_class=AssetClass.FOREX,
                pip_size=0.00001, decimal_places=5, quote_precision=5,  # Modern 5-decimal GBP/CHF
                min_lot_size=0.01, max_lot_size=1000.0, lot_step=0.01,
                pip_value_per_lot=1.08, contract_size=100000,  # ~$1.08 per pip per standard lot
                market_open_days=[0,1,2,3,4], market_open_hours=(0,24), is_24_7=True,
                margin_percentage=0.03, description="British Pound vs Swiss Franc",
                base_currency='GBP', quote_currency='CHF'
            ),
            'GBPCAD': InstrumentMetadata(
                symbol='GBPCAD', asset_class=AssetClass.FOREX,
                pip_size=0.00001, decimal_places=5, quote_precision=5,  # Modern 5-decimal GBP/CAD
                min_lot_size=0.01, max_lot_size=1000.0, lot_step=0.01,
                pip_value_per_lot=0.74, contract_size=100000,  # ~$0.74 per pip per standard lot
                market_open_days=[0,1,2,3,4], market_open_hours=(0,24), is_24_7=True,
                margin_percentage=0.03, description="British Pound vs Canadian Dollar",
                base_currency='GBP', quote_currency='CAD'
            ),
            
            # === JPY CROSS PAIRS ===
            'AUDJPY': InstrumentMetadata(
                symbol='AUDJPY', asset_class=AssetClass.FOREX,
                pip_size=0.001, decimal_places=3, quote_precision=3,  # Modern 3-decimal AUD/JPY
                min_lot_size=0.01, max_lot_size=1000.0, lot_step=0.01,
                pip_value_per_lot=0.67, contract_size=100000,  # ~$0.67 per pip per standard lot
                market_open_days=[0,1,2,3,4], market_open_hours=(0,24), is_24_7=True,
                margin_percentage=0.03, description="Australian Dollar vs Japanese Yen",
                base_currency='AUD', quote_currency='JPY'
            ),
            'CADJPY': InstrumentMetadata(
                symbol='CADJPY', asset_class=AssetClass.FOREX,
                pip_size=0.001, decimal_places=3, quote_precision=3,  # Modern 3-decimal CAD/JPY
                min_lot_size=0.01, max_lot_size=1000.0, lot_step=0.01,
                pip_value_per_lot=0.67, contract_size=100000,  # ~$0.67 per pip per standard lot
                market_open_days=[0,1,2,3,4], market_open_hours=(0,24), is_24_7=True,
                margin_percentage=0.03, description="Canadian Dollar vs Japanese Yen",
                base_currency='CAD', quote_currency='JPY'
            ),
            'CHFJPY': InstrumentMetadata(
                symbol='CHFJPY', asset_class=AssetClass.FOREX,
                pip_size=0.001, decimal_places=3, quote_precision=3,  # Modern 3-decimal CHF/JPY
                min_lot_size=0.01, max_lot_size=1000.0, lot_step=0.01,
                pip_value_per_lot=0.67, contract_size=100000,  # ~$0.67 per pip per standard lot
                market_open_days=[0,1,2,3,4], market_open_hours=(0,24), is_24_7=True,
                margin_percentage=0.03, description="Swiss Franc vs Japanese Yen",
                base_currency='CHF', quote_currency='JPY'
            ),
            'NZDJPY': InstrumentMetadata(
                symbol='NZDJPY', asset_class=AssetClass.FOREX,
                pip_size=0.001, decimal_places=3, quote_precision=3,  # Modern 3-decimal NZD/JPY
                min_lot_size=0.01, max_lot_size=1000.0, lot_step=0.01,
                pip_value_per_lot=0.67, contract_size=100000,  # ~$0.67 per pip per standard lot
                market_open_days=[0,1,2,3,4], market_open_hours=(0,24), is_24_7=True,
                margin_percentage=0.03, description="New Zealand Dollar vs Japanese Yen",
                base_currency='NZD', quote_currency='JPY'
            ),
            
            # === OTHER MAJOR CROSS PAIRS ===
            'AUDCAD': InstrumentMetadata(
                symbol='AUDCAD', asset_class=AssetClass.FOREX,
                pip_size=0.00001, decimal_places=5, quote_precision=5,  # Modern 5-decimal AUD/CAD
                min_lot_size=0.01, max_lot_size=1000.0, lot_step=0.01,
                pip_value_per_lot=0.74, contract_size=100000,  # ~$0.74 per pip per standard lot
                market_open_days=[0,1,2,3,4], market_open_hours=(0,24), is_24_7=True,
                margin_percentage=0.03, description="Australian Dollar vs Canadian Dollar",
                base_currency='AUD', quote_currency='CAD'
            ),
            'AUDCHF': InstrumentMetadata(
                symbol='AUDCHF', asset_class=AssetClass.FOREX,
                pip_size=0.00001, decimal_places=5, quote_precision=5,  # Modern 5-decimal AUD/CHF
                min_lot_size=0.01, max_lot_size=1000.0, lot_step=0.01,
                pip_value_per_lot=1.08, contract_size=100000,  # ~$1.08 per pip per standard lot
                market_open_days=[0,1,2,3,4], market_open_hours=(0,24), is_24_7=True,
                margin_percentage=0.03, description="Australian Dollar vs Swiss Franc",
                base_currency='AUD', quote_currency='CHF'
            ),
            'AUDNZD': InstrumentMetadata(
                symbol='AUDNZD', asset_class=AssetClass.FOREX,
                pip_size=0.00001, decimal_places=5, quote_precision=5,  # Modern 5-decimal AUD/NZD
                min_lot_size=0.01, max_lot_size=1000.0, lot_step=0.01,
                pip_value_per_lot=0.62, contract_size=100000,  # ~$0.62 per pip per standard lot
                market_open_days=[0,1,2,3,4], market_open_hours=(0,24), is_24_7=True,
                margin_percentage=0.03, description="Australian Dollar vs New Zealand Dollar",
                base_currency='AUD', quote_currency='NZD'
            ),
            'CADCHF': InstrumentMetadata(
                symbol='CADCHF', asset_class=AssetClass.FOREX,
                pip_size=0.00001, decimal_places=5, quote_precision=5,  # Modern 5-decimal CAD/CHF
                min_lot_size=0.01, max_lot_size=1000.0, lot_step=0.01,
                pip_value_per_lot=1.08, contract_size=100000,  # ~$1.08 per pip per standard lot
                market_open_days=[0,1,2,3,4], market_open_hours=(0,24), is_24_7=True,
                margin_percentage=0.03, description="Canadian Dollar vs Swiss Franc",
                base_currency='CAD', quote_currency='CHF'
            ),
            'NZDCAD': InstrumentMetadata(
                symbol='NZDCAD', asset_class=AssetClass.FOREX,
                pip_size=0.00001, decimal_places=5, quote_precision=5,  # Modern 5-decimal NZD/CAD
                min_lot_size=0.01, max_lot_size=1000.0, lot_step=0.01,
                pip_value_per_lot=0.74, contract_size=100000,  # ~$0.74 per pip per standard lot
                market_open_days=[0,1,2,3,4], market_open_hours=(0,24), is_24_7=True,
                margin_percentage=0.03, description="New Zealand Dollar vs Canadian Dollar",
                base_currency='NZD', quote_currency='CAD'
            ),
            'NZDCHF': InstrumentMetadata(
                symbol='NZDCHF', asset_class=AssetClass.FOREX,
                pip_size=0.00001, decimal_places=5, quote_precision=5,  # Modern 5-decimal NZD/CHF
                min_lot_size=0.01, max_lot_size=1000.0, lot_step=0.01,
                pip_value_per_lot=1.08, contract_size=100000,  # ~$1.08 per pip per standard lot
                market_open_days=[0,1,2,3,4], market_open_hours=(0,24), is_24_7=True,
                margin_percentage=0.03, description="New Zealand Dollar vs Swiss Franc",
                base_currency='NZD', quote_currency='CHF'
            )
        }
        
        # === MAJOR CRYPTOCURRENCY PAIRS ===
        # Professional crypto trading specifications with modern precision and 24/7 markets
        # All crypto markets are 24/7 with higher volatility margin requirements
        crypto_pairs = {
            'BTCUSD': InstrumentMetadata(
                symbol='BTCUSD', asset_class=AssetClass.CRYPTO,
                pip_size=0.01, decimal_places=2, quote_precision=2,  # Bitcoin precision: $0.01
                min_lot_size=0.001, max_lot_size=100.0, lot_step=0.001,
                pip_value_per_lot=0.01, contract_size=1,  # Direct 1:1 contract size for crypto
                market_open_days=[0,1,2,3,4,5,6], market_open_hours=(0,24), is_24_7=True,  # 24/7 crypto market
                margin_percentage=0.05, description="Bitcoin vs US Dollar",
                base_currency='BTC', quote_currency='USD'
            ),
            'ETHUSD': InstrumentMetadata(
                symbol='ETHUSD', asset_class=AssetClass.CRYPTO,
                pip_size=0.01, decimal_places=2, quote_precision=2,  # Ethereum precision: $0.01
                min_lot_size=0.01, max_lot_size=1000.0, lot_step=0.01,
                pip_value_per_lot=0.01, contract_size=1,  # Direct 1:1 contract size for crypto
                market_open_days=[0,1,2,3,4,5,6], market_open_hours=(0,24), is_24_7=True,  # 24/7 crypto market
                margin_percentage=0.05, description="Ethereum vs US Dollar",
                base_currency='ETH', quote_currency='USD'
            ),
            'ADAUSD': InstrumentMetadata(
                symbol='ADAUSD', asset_class=AssetClass.CRYPTO,
                pip_size=0.0001, decimal_places=4, quote_precision=4,  # Cardano precision: $0.0001
                min_lot_size=1.0, max_lot_size=100000.0, lot_step=1.0,
                pip_value_per_lot=0.0001, contract_size=1,  # Direct 1:1 contract size for crypto
                market_open_days=[0,1,2,3,4,5,6], market_open_hours=(0,24), is_24_7=True,  # 24/7 crypto market
                margin_percentage=0.07, description="Cardano vs US Dollar",
                base_currency='ADA', quote_currency='USD'
            ),
            'DOGEUSD': InstrumentMetadata(
                symbol='DOGEUSD', asset_class=AssetClass.CRYPTO,
                pip_size=0.00001, decimal_places=5, quote_precision=5,  # Dogecoin precision: $0.00001
                min_lot_size=10.0, max_lot_size=1000000.0, lot_step=10.0,
                pip_value_per_lot=0.00001, contract_size=1,  # Direct 1:1 contract size for crypto
                market_open_days=[0,1,2,3,4,5,6], market_open_hours=(0,24), is_24_7=True,  # 24/7 crypto market
                margin_percentage=0.10, description="Dogecoin vs US Dollar",
                base_currency='DOGE', quote_currency='USD'
            ),
            'SOLUSD': InstrumentMetadata(
                symbol='SOLUSD', asset_class=AssetClass.CRYPTO,
                pip_size=0.001, decimal_places=3, quote_precision=3,  # Solana precision: $0.001
                min_lot_size=0.1, max_lot_size=10000.0, lot_step=0.1,
                pip_value_per_lot=0.001, contract_size=1,  # Direct 1:1 contract size for crypto
                market_open_days=[0,1,2,3,4,5,6], market_open_hours=(0,24), is_24_7=True,  # 24/7 crypto market
                margin_percentage=0.08, description="Solana vs US Dollar",
                base_currency='SOL', quote_currency='USD'
            ),
            'BNBUSD': InstrumentMetadata(
                symbol='BNBUSD', asset_class=AssetClass.CRYPTO,
                pip_size=0.01, decimal_places=2, quote_precision=2,  # Binance Coin precision: $0.01
                min_lot_size=0.01, max_lot_size=1000.0, lot_step=0.01,
                pip_value_per_lot=0.01, contract_size=1,  # Direct 1:1 contract size for crypto
                market_open_days=[0,1,2,3,4,5,6], market_open_hours=(0,24), is_24_7=True,  # 24/7 crypto market
                margin_percentage=0.06, description="Binance Coin vs US Dollar",
                base_currency='BNB', quote_currency='USD'
            ),
            'XRPUSD': InstrumentMetadata(
                symbol='XRPUSD', asset_class=AssetClass.CRYPTO,
                pip_size=0.0001, decimal_places=4, quote_precision=4,  # Ripple precision: $0.0001
                min_lot_size=1.0, max_lot_size=100000.0, lot_step=1.0,
                pip_value_per_lot=0.0001, contract_size=1,  # Direct 1:1 contract size for crypto
                market_open_days=[0,1,2,3,4,5,6], market_open_hours=(0,24), is_24_7=True,  # 24/7 crypto market
                margin_percentage=0.08, description="Ripple vs US Dollar",
                base_currency='XRP', quote_currency='USD'
            ),
            'MATICUSD': InstrumentMetadata(
                symbol='MATICUSD', asset_class=AssetClass.CRYPTO,
                pip_size=0.0001, decimal_places=4, quote_precision=4,  # Polygon precision: $0.0001
                min_lot_size=1.0, max_lot_size=100000.0, lot_step=1.0,
                pip_value_per_lot=0.0001, contract_size=1,  # Direct 1:1 contract size for crypto
                market_open_days=[0,1,2,3,4,5,6], market_open_hours=(0,24), is_24_7=True,  # 24/7 crypto market
                margin_percentage=0.09, description="Polygon vs US Dollar",
                base_currency='MATIC', quote_currency='USD'
            )
        }
        
        # === COMMODITY INSTRUMENTS ===
        # Professional precious metals and oil trading specifications
        # Commodities have extended market hours but close on weekends
        commodity_pairs = {
            'XAUUSD': InstrumentMetadata(
                symbol='XAUUSD', asset_class=AssetClass.METALS,
                pip_size=0.01, decimal_places=2, quote_precision=2,  # Gold precision: $0.01 per troy ounce
                min_lot_size=0.01, max_lot_size=100.0, lot_step=0.01,
                pip_value_per_lot=0.01, contract_size=100,  # 100 troy ounces per lot
                market_open_days=[0,1,2,3,4], market_open_hours=(22,21), is_24_7=False,  # Sun 22:00 - Fri 21:00 UTC
                margin_percentage=0.05, description="Gold vs US Dollar",
                base_currency='XAU', quote_currency='USD'
            ),
            'XAGUSD': InstrumentMetadata(
                symbol='XAGUSD', asset_class=AssetClass.METALS,
                pip_size=0.001, decimal_places=3, quote_precision=3,  # Silver precision: $0.001 per troy ounce
                min_lot_size=0.01, max_lot_size=500.0, lot_step=0.01,
                pip_value_per_lot=0.001, contract_size=5000,  # 5000 troy ounces per lot
                market_open_days=[0,1,2,3,4], market_open_hours=(22,21), is_24_7=False,  # Sun 22:00 - Fri 21:00 UTC
                margin_percentage=0.07, description="Silver vs US Dollar",
                base_currency='XAG', quote_currency='USD'
            ),
            'USOIL': InstrumentMetadata(
                symbol='USOIL', asset_class=AssetClass.OIL,
                pip_size=0.01, decimal_places=2, quote_precision=2,  # WTI Crude Oil precision: $0.01 per barrel
                min_lot_size=0.01, max_lot_size=100.0, lot_step=0.01,
                pip_value_per_lot=0.01, contract_size=1000,  # 1000 barrels per lot
                market_open_days=[0,1,2,3,4], market_open_hours=(22,21), is_24_7=False,  # Sun 22:00 - Fri 21:00 UTC
                margin_percentage=0.10, description="WTI Crude Oil",
                base_currency='OIL', quote_currency='USD'
            )
        }
        
        # Combine forex, crypto, and commodity instruments for comprehensive trading coverage
        instruments.update(forex_pairs)
        instruments.update(crypto_pairs)
        instruments.update(commodity_pairs)
        
        return instruments
    
    def get_instrument(self, symbol: str) -> Optional[InstrumentMetadata]:
        """Get instrument metadata by symbol"""
        return self._instruments.get(symbol.upper())
    
    def get_pip_size(self, symbol: str) -> float:
        """Get pip/tick size for a symbol"""
        instrument = self.get_instrument(symbol)
        return instrument.pip_size if instrument else 0.0001  # Default fallback
    
    def get_decimal_places(self, symbol: str) -> int:
        """Get decimal places for price display"""
        instrument = self.get_instrument(symbol)
        return instrument.decimal_places if instrument else 5  # Default fallback
    
    def get_pip_value_per_lot(self, symbol: str) -> float:
        """Get pip value per lot in USD"""
        instrument = self.get_instrument(symbol)
        return instrument.pip_value_per_lot if instrument else 10.0  # Default fallback
    
    def get_min_lot_size(self, symbol: str) -> float:
        """Get minimum lot size"""
        instrument = self.get_instrument(symbol)
        return instrument.min_lot_size if instrument else 0.01  # Default fallback
    
    def get_max_lot_size(self, symbol: str) -> float:
        """Get maximum lot size"""
        instrument = self.get_instrument(symbol)
        return instrument.max_lot_size if instrument else 100.0  # Default fallback
    
    def get_lot_step(self, symbol: str) -> float:
        """Get lot size increment"""
        instrument = self.get_instrument(symbol)
        return instrument.lot_step if instrument else 0.01  # Default fallback
    
    def get_asset_class(self, symbol: str) -> AssetClass:
        """Get asset class for a symbol"""
        instrument = self.get_instrument(symbol)
        return instrument.asset_class if instrument else AssetClass.FOREX  # Default fallback
    
    def is_market_open_24_7(self, symbol: str) -> bool:
        """Check if market is open 24/7"""
        instrument = self.get_instrument(symbol)
        return instrument.is_24_7 if instrument else False
    
    def get_margin_percentage(self, symbol: str) -> float:
        """Get margin requirement percentage"""
        instrument = self.get_instrument(symbol)
        return instrument.margin_percentage if instrument else 0.01  # Default 1%
    
    def format_price(self, symbol: str, price: float) -> str:
        """Format price according to instrument specifications"""
        decimal_places = self.get_decimal_places(symbol)
        return f"{price:.{decimal_places}f}"
    
    def round_lot_size(self, symbol: str, lot_size: float) -> float:
        """Round lot size to valid increment"""
        step = self.get_lot_step(symbol)
        min_size = self.get_min_lot_size(symbol)
        max_size = self.get_max_lot_size(symbol)
        
        # Round to nearest step
        rounded = round(lot_size / step) * step
        
        # Apply min/max constraints
        rounded = max(min_size, min(max_size, rounded))
        
        return round(rounded, 3)  # Round to 3 decimal places for precision
    
    def get_all_symbols(self) -> list:
        """Get list of all available symbols"""
        return list(self._instruments.keys())
    
    def get_symbols_by_asset_class(self, asset_class: AssetClass) -> list:
        """Get symbols filtered by asset class"""
        return [
            symbol for symbol, metadata in self._instruments.items()
            if metadata.asset_class == asset_class
        ]

# Global singleton instance
instrument_db = InstrumentMetadataDB()

# Convenience functions for easy access
def get_instrument_metadata(symbol: str) -> Optional[InstrumentMetadata]:
    """Get complete instrument metadata"""
    return instrument_db.get_instrument(symbol)

def get_pip_size(symbol: str) -> float:
    """Get pip/tick size for accurate price calculations"""
    return instrument_db.get_pip_size(symbol)

def get_pip_value_per_lot(symbol: str) -> float:
    """Get pip value per lot for position sizing"""
    return instrument_db.get_pip_value_per_lot(symbol)

def format_price(symbol: str, price: float) -> str:
    """Format price with correct decimal places"""
    return instrument_db.format_price(symbol, price)

def round_lot_size(symbol: str, lot_size: float) -> float:
    """Round lot size to valid trading increment"""
    return instrument_db.round_lot_size(symbol, lot_size)

def get_asset_class(symbol: str) -> AssetClass:
    """Get asset class classification"""
    return instrument_db.get_asset_class(symbol)