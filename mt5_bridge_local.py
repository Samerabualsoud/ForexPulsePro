"""
MT5 Bridge Service - Local Windows Installation
Connects ACY Securities MT5 terminal to Replit Forex Signal Dashboard
"""
import os
import sys
import time
import json
import logging
import hashlib
import hmac
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from threading import Thread
import asyncio

# External libraries (install via pip)
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import MetaTrader5 as mt5

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mt5_bridge.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuration from Replit secrets
MT5_LOGIN = 841946
MT5_PASSWORD = "2141991@Forex"
MT5_SERVER = "ACYSecurities-Demo"
MT5_BRIDGE_SECRET = "c8614671c7d00af6bcbc578a6ce7328cc888f68b89d314a2daf4e59507c65f8b"
BRIDGE_PORT = 8001

# FastAPI app setup
app = FastAPI(title="MT5 Bridge Service", version="1.0.0")
security = HTTPBearer()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global MT5 connection state
mt5_connected = False
account_info = {}

@dataclass
class OrderRequest:
    """Order execution request"""
    symbol: str
    order_type: str  # MARKET_BUY, MARKET_SELL, LIMIT_BUY, LIMIT_SELL
    volume: float
    price: Optional[float] = None
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
    status: str = "PENDING"
    message: Optional[str] = None
    error_code: Optional[int] = None
    execution_time: Optional[str] = None
    slippage: Optional[float] = None

def verify_signature(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify the bridge secret"""
    token = credentials.credentials
    if not hmac.compare_digest(token, MT5_BRIDGE_SECRET):
        raise HTTPException(status_code=401, detail="Invalid bridge secret")
    return True

def initialize_mt5():
    """Initialize MT5 connection"""
    global mt5_connected, account_info
    
    try:
        # Initialize MT5 connection
        if not mt5.initialize():
            logger.error(f"MT5 initialize failed, error code: {mt5.last_error()}")
            return False
        
        # Login to trading account
        if not mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
            logger.error(f"MT5 login failed, error code: {mt5.last_error()}")
            mt5.shutdown()
            return False
        
        # Get account info
        account = mt5.account_info()
        if account is None:
            logger.error(f"Failed to get account info, error code: {mt5.last_error()}")
            mt5.shutdown()
            return False
        
        account_info = {
            "login": account.login,
            "trade_mode": account.trade_mode,
            "balance": account.balance,
            "equity": account.equity,
            "margin": account.margin,
            "margin_free": account.margin_free,
            "margin_level": account.margin_level,
            "currency": account.currency,
            "server": account.server,
            "company": account.company
        }
        
        mt5_connected = True
        logger.info(f"MT5 connected successfully - Account: {account.login}, Balance: {account.balance} {account.currency}")
        return True
        
    except Exception as e:
        logger.error(f"MT5 initialization error: {str(e)}")
        return False

def get_symbol_info(symbol: str):
    """Get symbol information"""
    try:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return None
        
        return {
            "symbol": symbol_info.name,
            "bid": symbol_info.bid,
            "ask": symbol_info.ask,
            "spread": symbol_info.spread,
            "digits": symbol_info.digits,
            "point": symbol_info.point,
            "trade_mode": symbol_info.trade_mode,
            "volume_min": symbol_info.volume_min,
            "volume_max": symbol_info.volume_max,
            "volume_step": symbol_info.volume_step
        }
    except Exception as e:
        logger.error(f"Error getting symbol info for {symbol}: {str(e)}")
        return None

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global mt5_connected
    
    if not mt5_connected:
        # Try to reconnect
        mt5_connected = initialize_mt5()
    
    return {
        "status": "healthy" if mt5_connected else "unhealthy",
        "mt5_connected": mt5_connected,
        "account": account_info.get("login") if mt5_connected else None,
        "server": account_info.get("server") if mt5_connected else None,
        "balance": account_info.get("balance") if mt5_connected else None,
        "currency": account_info.get("currency") if mt5_connected else None,
        "terminal_connected": mt5.terminal_info().connected if mt5_connected else False,
        "local_time": datetime.now().isoformat(),
        "bridge_version": "1.0.0"
    }

@app.get("/account")
async def get_account_info(authorized: bool = Depends(verify_signature)):
    """Get account information"""
    if not mt5_connected:
        raise HTTPException(status_code=503, detail="MT5 not connected")
    
    try:
        account = mt5.account_info()
        if account is None:
            raise HTTPException(status_code=500, detail="Failed to get account info")
        
        return {
            "login": account.login,
            "trade_mode": account.trade_mode,
            "balance": account.balance,
            "equity": account.equity,
            "margin": account.margin,
            "margin_free": account.margin_free,
            "margin_level": account.margin_level,
            "profit": account.profit,
            "currency": account.currency,
            "server": account.server,
            "company": account.company
        }
    except Exception as e:
        logger.error(f"Error getting account info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/positions")
async def get_positions(symbol: Optional[str] = None, authorized: bool = Depends(verify_signature)):
    """Get open positions"""
    if not mt5_connected:
        raise HTTPException(status_code=503, detail="MT5 not connected")
    
    try:
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()
        
        if positions is None:
            return []
        
        result = []
        for pos in positions:
            result.append({
                "ticket": pos.ticket,
                "symbol": pos.symbol,
                "type": pos.type,
                "volume": pos.volume,
                "price_open": pos.price_open,
                "price_current": pos.price_current,
                "profit": pos.profit,
                "swap": pos.swap,
                "comment": pos.comment,
                "magic": pos.magic,
                "time": pos.time
            })
        
        return result
    except Exception as e:
        logger.error(f"Error getting positions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/symbol/{symbol}")
async def get_symbol_data(symbol: str, authorized: bool = Depends(verify_signature)):
    """Get symbol information and current price"""
    if not mt5_connected:
        raise HTTPException(status_code=503, detail="MT5 not connected")
    
    try:
        symbol_info = get_symbol_info(symbol)
        if symbol_info is None:
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")
        
        # Get latest tick
        tick = mt5.symbol_info_tick(symbol)
        if tick is not None:
            symbol_info.update({
                "bid": tick.bid,
                "ask": tick.ask,
                "last": tick.last,
                "volume": tick.volume,
                "time": tick.time
            })
        
        return symbol_info
    except Exception as e:
        logger.error(f"Error getting symbol data for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/execute")
async def execute_order(order: OrderRequest, authorized: bool = Depends(verify_signature)):
    """Execute a trading order"""
    if not mt5_connected:
        raise HTTPException(status_code=503, detail="MT5 not connected")
    
    try:
        # Get symbol info
        symbol_info = get_symbol_info(order.symbol)
        if symbol_info is None:
            return ExecutionResult(
                success=False,
                message=f"Symbol {order.symbol} not found",
                error_code=10004
            )
        
        # Prepare order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": order.symbol,
            "volume": order.volume,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Set order type
        if order.order_type == "MARKET_BUY":
            request["type"] = mt5.ORDER_TYPE_BUY
            request["price"] = symbol_info["ask"]
        elif order.order_type == "MARKET_SELL":
            request["type"] = mt5.ORDER_TYPE_SELL
            request["price"] = symbol_info["bid"]
        elif order.order_type == "LIMIT_BUY":
            request["type"] = mt5.ORDER_TYPE_BUY_LIMIT
            request["price"] = order.price
        elif order.order_type == "LIMIT_SELL":
            request["type"] = mt5.ORDER_TYPE_SELL_LIMIT
            request["price"] = order.price
        else:
            return ExecutionResult(
                success=False,
                message=f"Invalid order type: {order.order_type}",
                error_code=10015
            )
        
        # Add optional parameters
        if order.stop_loss:
            request["sl"] = order.stop_loss
        if order.take_profit:
            request["tp"] = order.take_profit
        if order.comment:
            request["comment"] = order.comment
        if order.magic_number:
            request["magic"] = order.magic_number
        
        # Execute the order
        result = mt5.order_send(request)
        
        if result is None:
            return ExecutionResult(
                success=False,
                message="Order execution failed - no result",
                error_code=mt5.last_error()[0]
            )
        
        # Process result
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            execution_time = datetime.now(timezone.utc).isoformat()
            slippage = abs(result.price - request["price"]) if result.price and request.get("price") else None
            
            return ExecutionResult(
                success=True,
                order_id=str(result.order),
                ticket=str(result.order),
                executed_price=result.price,
                executed_volume=result.volume,
                status="FILLED",
                message="Order executed successfully",
                execution_time=execution_time,
                slippage=slippage
            )
        else:
            return ExecutionResult(
                success=False,
                message=f"Order execution failed: {result.comment}",
                error_code=result.retcode,
                status="REJECTED"
            )
            
    except Exception as e:
        logger.error(f"Error executing order: {str(e)}")
        return ExecutionResult(
            success=False,
            message=f"Execution error: {str(e)}",
            error_code=10001,
            status="REJECTED"
        )

@app.post("/close/{ticket}")
async def close_position(ticket: str, authorized: bool = Depends(verify_signature)):
    """Close a specific position"""
    if not mt5_connected:
        raise HTTPException(status_code=503, detail="MT5 not connected")
    
    try:
        # Get position info
        position = mt5.positions_get(ticket=int(ticket))
        if not position:
            return ExecutionResult(
                success=False,
                message=f"Position {ticket} not found",
                error_code=10004
            )
        
        pos = position[0]
        
        # Prepare close request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "position": pos.ticket,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
            "comment": "Closed by signal dashboard"
        }
        
        # Get current price
        symbol_info = get_symbol_info(pos.symbol)
        if symbol_info:
            request["price"] = symbol_info["bid"] if pos.type == mt5.ORDER_TYPE_BUY else symbol_info["ask"]
        
        # Close position
        result = mt5.order_send(request)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            return ExecutionResult(
                success=True,
                ticket=str(result.order),
                executed_price=result.price,
                executed_volume=result.volume,
                status="FILLED",
                message="Position closed successfully",
                execution_time=datetime.now(timezone.utc).isoformat()
            )
        else:
            return ExecutionResult(
                success=False,
                message=f"Failed to close position: {result.comment if result else 'Unknown error'}",
                error_code=result.retcode if result else mt5.last_error()[0],
                status="REJECTED"
            )
            
    except Exception as e:
        logger.error(f"Error closing position {ticket}: {str(e)}")
        return ExecutionResult(
            success=False,
            message=f"Close error: {str(e)}",
            error_code=10001,
            status="REJECTED"
        )

def monitor_connection():
    """Monitor MT5 connection and reconnect if needed"""
    global mt5_connected
    
    while True:
        try:
            if mt5_connected:
                # Check if connection is still alive
                terminal_info = mt5.terminal_info()
                if terminal_info is None or not terminal_info.connected:
                    logger.warning("MT5 terminal disconnected, attempting reconnection...")
                    mt5_connected = False
                    initialize_mt5()
            else:
                # Try to reconnect
                logger.info("Attempting MT5 reconnection...")
                initialize_mt5()
            
            time.sleep(30)  # Check every 30 seconds
            
        except Exception as e:
            logger.error(f"Connection monitor error: {str(e)}")
            time.sleep(60)  # Wait longer on error

@app.on_event("startup")
async def startup_event():
    """Initialize MT5 on startup"""
    logger.info("Starting MT5 Bridge Service...")
    if initialize_mt5():
        logger.info("MT5 Bridge Service started successfully")
        # Start connection monitor in background
        Thread(target=monitor_connection, daemon=True).start()
    else:
        logger.error("Failed to initialize MT5 connection")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down MT5 Bridge Service...")
    if mt5_connected:
        mt5.shutdown()
    logger.info("MT5 Bridge Service stopped")

if __name__ == "__main__":
    print("=" * 60)
    print("MT5 Bridge Service - ACY Securities Integration")
    print("=" * 60)
    print(f"Account: {MT5_LOGIN}")
    print(f"Server: {MT5_SERVER}")
    print(f"Port: {BRIDGE_PORT}")
    print(f"Bridge Secret: {MT5_BRIDGE_SECRET[:10]}...")
    print("=" * 60)
    
    # Run the FastAPI server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=BRIDGE_PORT,
        log_level="info"
    )