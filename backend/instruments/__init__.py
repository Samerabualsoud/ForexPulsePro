"""
Instrument Metadata Package
Provides comprehensive metadata for forex, crypto, and metals trading instruments
"""

from .metadata import (
    instrument_db,
    AssetClass,
    InstrumentMetadata,
    get_instrument_metadata,
    get_pip_size,
    get_pip_value_per_lot,
    format_price,
    round_lot_size,
    get_asset_class
)

__all__ = [
    'instrument_db',
    'AssetClass', 
    'InstrumentMetadata',
    'get_instrument_metadata',
    'get_pip_size',
    'get_pip_value_per_lot',
    'format_price',
    'round_lot_size',
    'get_asset_class'
]