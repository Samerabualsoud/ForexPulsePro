"""
Base Execution Provider Interface
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class OrderType(Enum):
    MARKET_BUY = "MARKET_BUY"
    MARKET_SELL = "MARKET_SELL"
    LIMIT_BUY = "LIMIT_BUY"
    LIMIT_SELL = "LIMIT_SELL"

class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    PARTIAL = "PARTIAL"

@dataclass
class OrderRequest:
    """Order execution request"""
    symbol: str
    order_type: OrderType
    volume: float  # Lot size
    price: Optional[float] = None  # For limit orders
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    comment: Optional[str] = None
    magic_number: Optional[int] = None

@dataclass
class ExecutionResult:
    """Result of order execution"""
    success: bool
    order_id: Optional[str] = None
    ticket: Optional[str] = None
    executed_price: Optional[float] = None
    executed_volume: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    message: Optional[str] = None
    error_code: Optional[int] = None
    execution_time: Optional[datetime] = None
    slippage: Optional[float] = None

class ExecutionProvider(ABC):
    """Abstract base class for trade execution providers"""
    
    @abstractmethod
    async def execute_order(self, order_request: OrderRequest) -> ExecutionResult:
        """Execute a trading order"""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        pass
    
    @abstractmethod
    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open positions"""
        pass
    
    @abstractmethod
    async def close_position(self, ticket: str) -> ExecutionResult:
        """Close a specific position"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the execution provider is healthy"""
        pass