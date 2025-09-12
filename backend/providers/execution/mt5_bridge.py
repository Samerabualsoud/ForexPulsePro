"""
MT5 Bridge Execution Provider
Connects to Windows VPS MT5 Bridge Service via REST API
"""
import httpx
import hashlib
import hmac
import time
import json
import os
from typing import Dict, Any, Optional, List
from datetime import datetime

from .base import ExecutionProvider, ExecutionResult, OrderRequest, OrderType, OrderStatus
from ...logs.logger import get_logger

logger = get_logger(__name__)

class MT5BridgeExecutionProvider(ExecutionProvider):
    """MT5 Bridge execution provider for ACY Securities"""
    
    def __init__(self):
        self.bridge_url = os.getenv('MT5_BRIDGE_URL', 'https://your-mt5-bridge.tunnel.example.com')
        self.bridge_secret = os.getenv('MT5_BRIDGE_SECRET', '')
        self.timeout = 30.0
        self.retry_attempts = 3
        
        if not self.bridge_secret:
            logger.warning("MT5_BRIDGE_SECRET not configured - bridge authentication will fail")
    
    def _generate_signature(self, method: str, path: str, body: str = '') -> str:
        """Generate HMAC signature for request authentication"""
        timestamp = str(int(time.time()))
        message = f"{method.upper()}{path}{timestamp}{body}"
        signature = hmac.new(
            self.bridge_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return f"{timestamp}.{signature}"
    
    def _get_headers(self, method: str, path: str, body: str = '') -> Dict[str, str]:
        """Get authentication headers for bridge requests"""
        signature = self._generate_signature(method, path, body)
        return {
            'Content-Type': 'application/json',
            'X-MT5-Signature': signature,
            'User-Agent': 'ForexSignalDashboard/1.0'
        }
    
    async def _make_request(self, method: str, path: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make authenticated request to MT5 bridge"""
        body = json.dumps(data) if data else ''
        headers = self._get_headers(method, path, body)
        
        for attempt in range(self.retry_attempts):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.request(
                        method=method,
                        url=f"{self.bridge_url}{path}",
                        headers=headers,
                        content=body if body else None
                    )
                    
                    if response.status_code == 200:
                        return response.json()
                    elif response.status_code == 401:
                        raise Exception("Bridge authentication failed - check MT5_BRIDGE_SECRET")
                    elif response.status_code >= 500:
                        # Server error - retry
                        if attempt < self.retry_attempts - 1:
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                            continue
                        raise Exception(f"Bridge server error: {response.status_code}")
                    else:
                        raise Exception(f"Bridge request failed: {response.status_code} - {response.text}")
                        
            except httpx.TimeoutException:
                if attempt < self.retry_attempts - 1:
                    logger.warning(f"Bridge timeout (attempt {attempt + 1}/{self.retry_attempts})")
                    continue
                raise Exception("Bridge connection timeout")
            except Exception as e:
                if attempt < self.retry_attempts - 1:
                    logger.warning(f"Bridge request error (attempt {attempt + 1}/{self.retry_attempts}): {e}")
                    continue
                raise
        
        raise Exception("All bridge request attempts failed")
    
    async def execute_order(self, order_request: OrderRequest) -> ExecutionResult:
        """Execute trading order through MT5 bridge"""
        try:
            # Convert internal order request to bridge format
            bridge_order = {
                'symbol': order_request.symbol,
                'action': 'BUY' if order_request.order_type in [OrderType.MARKET_BUY, OrderType.LIMIT_BUY] else 'SELL',
                'order_type': 'MARKET' if order_request.order_type in [OrderType.MARKET_BUY, OrderType.MARKET_SELL] else 'LIMIT',
                'volume': order_request.volume,
                'price': order_request.price,
                'stop_loss': order_request.stop_loss,
                'take_profit': order_request.take_profit,
                'comment': order_request.comment or f"ForexDashboard-{int(time.time())}",
                'magic': order_request.magic_number or 234000
            }
            
            logger.info(f"Executing order: {bridge_order['action']} {bridge_order['volume']} {bridge_order['symbol']}")
            
            response = await self._make_request('POST', '/api/orders', bridge_order)
            
            # Parse bridge response
            if response.get('success'):
                return ExecutionResult(
                    success=True,
                    order_id=str(response.get('order_id')),
                    ticket=str(response.get('ticket')),
                    executed_price=response.get('price'),
                    executed_volume=response.get('volume'),
                    status=OrderStatus.FILLED if response.get('status') == 'FILLED' else OrderStatus.PENDING,
                    message=response.get('message', 'Order executed successfully'),
                    execution_time=datetime.utcnow(),
                    slippage=response.get('slippage', 0.0)
                )
            else:
                return ExecutionResult(
                    success=False,
                    message=response.get('error', 'Order execution failed'),
                    error_code=response.get('error_code'),
                    status=OrderStatus.REJECTED,
                    execution_time=datetime.utcnow()
                )
                
        except Exception as e:
            logger.error(f"MT5 order execution failed: {e}")
            return ExecutionResult(
                success=False,
                message=f"Bridge connection error: {str(e)}",
                status=OrderStatus.REJECTED,
                execution_time=datetime.utcnow()
            )
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get MT5 account information"""
        try:
            response = await self._make_request('GET', '/api/account')
            return response
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {'error': str(e)}
    
    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open positions"""
        try:
            path = '/api/positions'
            if symbol:
                path += f"?symbol={symbol}"
            
            response = await self._make_request('GET', path)
            return response.get('positions', [])
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    async def close_position(self, ticket: str) -> ExecutionResult:
        """Close specific position"""
        try:
            response = await self._make_request('DELETE', f'/api/positions/{ticket}')
            
            if response.get('success'):
                return ExecutionResult(
                    success=True,
                    order_id=str(response.get('order_id')),
                    ticket=ticket,
                    executed_price=response.get('close_price'),
                    executed_volume=response.get('volume'),
                    status=OrderStatus.FILLED,
                    message="Position closed successfully",
                    execution_time=datetime.utcnow()
                )
            else:
                return ExecutionResult(
                    success=False,
                    message=response.get('error', 'Failed to close position'),
                    status=OrderStatus.REJECTED,
                    execution_time=datetime.utcnow()
                )
                
        except Exception as e:
            logger.error(f"Failed to close position {ticket}: {e}")
            return ExecutionResult(
                success=False,
                message=f"Bridge error: {str(e)}",
                status=OrderStatus.REJECTED,
                execution_time=datetime.utcnow()
            )
    
    async def health_check(self) -> bool:
        """Check if MT5 bridge is healthy"""
        try:
            response = await self._make_request('GET', '/health')
            return response.get('status') == 'healthy' and response.get('mt5_connected', False)
        except Exception as e:
            logger.error(f"Bridge health check failed: {e}")
            return False