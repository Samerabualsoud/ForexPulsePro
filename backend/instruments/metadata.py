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
        
        # === CRYPTO ONLY CONFIGURATION ===
        # Removed all forex pairs, metals, and commodities as per requirements
        # Keeping only cryptocurrency vs USD pairs for focused crypto trading
        
        # === CRYPTOCURRENCY PAIRS ===
        # Using realistic pip sizes for crypto trading (typical moves = 10-50 pips)
        crypto_pairs = {
            'BTCUSD': InstrumentMetadata(
                symbol='BTCUSD', asset_class=AssetClass.CRYPTO,
                pip_size=2.0, decimal_places=2, quote_precision=2,  # $100 move = 50 pips
                min_lot_size=0.001, max_lot_size=100.0, lot_step=0.001,
                pip_value_per_lot=2.0, contract_size=1,
                market_open_days=[0,1,2,3,4,5,6], market_open_hours=(0,24), is_24_7=True,
                margin_percentage=0.10, description="Bitcoin vs US Dollar",
                base_currency='BTC', quote_currency='USD'
            ),
            'ETHUSD': InstrumentMetadata(
                symbol='ETHUSD', asset_class=AssetClass.CRYPTO,
                pip_size=0.20, decimal_places=3, quote_precision=3,  # $10 move = 50 pips
                min_lot_size=0.01, max_lot_size=1000.0, lot_step=0.01,
                pip_value_per_lot=0.20, contract_size=1,
                market_open_days=[0,1,2,3,4,5,6], market_open_hours=(0,24), is_24_7=True,
                margin_percentage=0.10, description="Ethereum vs US Dollar",
                base_currency='ETH', quote_currency='USD'
            ),
            'LTCUSD': InstrumentMetadata(
                symbol='LTCUSD', asset_class=AssetClass.CRYPTO,
                pip_size=0.10, decimal_places=3, quote_precision=3,  # $5 move = 50 pips
                min_lot_size=0.1, max_lot_size=1000.0, lot_step=0.1,
                pip_value_per_lot=0.10, contract_size=1,
                market_open_days=[0,1,2,3,4,5,6], market_open_hours=(0,24), is_24_7=True,
                margin_percentage=0.10, description="Litecoin vs US Dollar",
                base_currency='LTC', quote_currency='USD'
            ),
            'ADAUSD': InstrumentMetadata(
                symbol='ADAUSD', asset_class=AssetClass.CRYPTO,
                pip_size=0.005, decimal_places=4, quote_precision=4,  # $0.25 move = 50 pips
                min_lot_size=1.0, max_lot_size=100000.0, lot_step=1.0,
                pip_value_per_lot=0.005, contract_size=1,
                market_open_days=[0,1,2,3,4,5,6], market_open_hours=(0,24), is_24_7=True,
                margin_percentage=0.10, description="Cardano vs US Dollar",
                base_currency='ADA', quote_currency='USD'
            ),
            'SOLUSD': InstrumentMetadata(
                symbol='SOLUSD', asset_class=AssetClass.CRYPTO,
                pip_size=0.50, decimal_places=3, quote_precision=3,  # $25 move = 50 pips
                min_lot_size=0.01, max_lot_size=1000.0, lot_step=0.01,
                pip_value_per_lot=0.50, contract_size=1,
                market_open_days=[0,1,2,3,4,5,6], market_open_hours=(0,24), is_24_7=True,
                margin_percentage=0.10, description="Solana vs US Dollar",
                base_currency='SOL', quote_currency='USD'
            )
        }
        
        # All metals and commodities removed for crypto-only configuration
        
        # Combine only cryptocurrency instruments
        instruments.update(crypto_pairs)
        
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