# MT5 Bridge Setup Guide for ACY Securities

## Overview
Your Forex Signal Dashboard now has MT5 integration capability that will:
- Get real market data directly from your MT5 terminal
- Execute trades automatically when signal confidence ≥ 85%
- Replace Alpha Vantage with live ACY Securities data

## Architecture
```
ACY Securities MT5 ← → Windows VPS Bridge ← → Replit Dashboard
                        (Python Service)      (Your App)
```

## Step 1: Windows VPS Setup

### 1.1 Get Windows VPS
- **Recommended**: AWS EC2 Windows Server or Azure Windows VM
- **Minimum**: 2 vCPU, 4GB RAM, 50GB SSD
- **Location**: Close to ACY Securities servers for low latency

### 1.2 Install MT5 Terminal
1. Download MT5 from ACY Securities
2. Login with your trading account credentials
3. Ensure market data is streaming properly

### 1.3 Install Python & Dependencies
```bash
# Download Python 3.11+ for Windows
pip install MetaTrader5 fastapi uvicorn pandas numpy python-jose cryptography
```

## Step 2: Bridge Service Code

Create this Python service on your Windows VPS:

**`mt5_bridge.py`**
```python
import MetaTrader5 as mt5
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import hashlib
import hmac
import time
import json
from datetime import datetime
from typing import Optional
import pandas as pd

app = FastAPI(title="MT5 Bridge for Forex Dashboard")

# Configuration
BRIDGE_SECRET = "your-super-secret-key-here"  # Change this!
MT5_LOGIN = 12345678  # Your ACY account number
MT5_PASSWORD = "your-mt5-password"
MT5_SERVER = "ACYSecurities-Demo"  # or Live server

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-replit-app-url.replit.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def verify_signature(signature: str, method: str, path: str, body: str):
    """Verify HMAC signature from Replit"""
    try:
        timestamp, sig = signature.split('.')
        message = f"{method.upper()}{path}{timestamp}{body}"
        expected = hmac.new(
            BRIDGE_SECRET.encode(), message.encode(), hashlib.sha256
        ).hexdigest()
        
        # Check timestamp (5 minute window)
        if abs(int(timestamp) - int(time.time())) > 300:
            return False
            
        return hmac.compare_digest(sig, expected)
    except:
        return False

def require_auth(x_mt5_signature: str = Header(...)):
    if not verify_signature(x_mt5_signature, "GET", "/", ""):
        raise HTTPException(status_code=401, detail="Invalid signature")

# Initialize MT5 connection
@app.on_event("startup")
async def startup():
    if not mt5.initialize():
        print(f"MT5 initialization failed: {mt5.last_error()}")
        return
    
    if not mt5.login(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER):
        print(f"MT5 login failed: {mt5.last_error()}")
        return
        
    print("MT5 Bridge started successfully!")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        account_info = mt5.account_info()
        return {
            "status": "healthy",
            "mt5_connected": account_info is not None,
            "server": account_info.server if account_info else None,
            "balance": account_info.balance if account_info else None
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/api/account")
async def get_account(auth=Depends(require_auth)):
    """Get account information"""
    account_info = mt5.account_info()
    if not account_info:
        raise HTTPException(status_code=503, detail="MT5 not connected")
    
    return {
        "login": account_info.login,
        "server": account_info.server,
        "balance": account_info.balance,
        "equity": account_info.equity,
        "margin": account_info.margin,
        "free_margin": account_info.margin_free,
        "currency": account_info.currency
    }

@app.get("/api/ohlc/{symbol}")
async def get_ohlc(symbol: str, count: int = 100):
    """Get OHLC data for symbol"""
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, count)
    if rates is None or len(rates) == 0:
        raise HTTPException(status_code=404, detail=f"No data for {symbol}")
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    return {
        "symbol": symbol,
        "data": df.to_dict('records')
    }

@app.post("/api/orders")
async def place_order(order_data: dict, auth=Depends(require_auth)):
    """Execute trading order"""
    try:
        symbol = order_data['symbol']
        action = order_data['action']  # BUY/SELL
        volume = order_data['volume']
        
        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            raise HTTPException(status_code=400, detail=f"No tick data for {symbol}")
        
        price = tick.ask if action == 'BUY' else tick.bid
        order_type = mt5.ORDER_TYPE_BUY if action == 'BUY' else mt5.ORDER_TYPE_SELL
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": order_data.get('stop_loss'),
            "tp": order_data.get('take_profit'),
            "deviation": 20,
            "magic": order_data.get('magic', 234000),
            "comment": order_data.get('comment', 'ForexDashboard'),
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            return {
                "success": True,
                "ticket": result.order,
                "price": result.price,
                "volume": result.volume,
                "status": "FILLED"
            }
        else:
            return {
                "success": False,
                "error": f"Order failed: {result.retcode}",
                "error_code": result.retcode
            }
            
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

## Step 3: Environment Configuration

### 3.1 Update Replit Secrets
Add these environment variables in your Replit project:

```env
# MT5 Bridge Configuration
MT5_BRIDGE_URL=https://your-vps-ip:8001
MT5_BRIDGE_SECRET=your-super-secret-key-here

# Auto-Trading Configuration  
AUTO_TRADE_ENABLED=true
AUTO_TRADE_CONFIDENCE_THRESHOLD=0.85
AUTO_TRADE_LOT_SIZE=0.01
```

### 3.2 Secure the Bridge
1. **Use HTTPS**: Set up Cloudflare Tunnel or SSL certificate
2. **Firewall**: Only allow connections from your Replit app
3. **Strong Secret**: Generate a 64-character random secret key

## Step 4: Testing

### 4.1 Test Bridge Connection
```python
# Run this in your Replit console
import requests
import hmac
import hashlib
import time

def test_bridge():
    bridge_url = "your-bridge-url"
    secret = "your-secret"
    
    # Test health endpoint
    response = requests.get(f"{bridge_url}/health")
    print("Health:", response.json())
    
    # Test authenticated endpoint
    timestamp = str(int(time.time()))
    message = f"GET/api/account{timestamp}"
    signature = hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()
    headers = {'X-MT5-Signature': f'{timestamp}.{signature}'}
    
    response = requests.get(f"{bridge_url}/api/account", headers=headers)
    print("Account:", response.json())

test_bridge()
```

## Step 5: Enable Auto-Trading

Once everything is connected:

1. **Start with Demo**: Test with ACY demo account first
2. **Monitor Signals**: Watch the dashboard for "Auto-Traded" signals
3. **Check MT5**: Verify trades appear in your MT5 terminal
4. **Go Live**: Switch to live account when comfortable

## Dashboard Features

Your dashboard will now show:
- 🟢 **Live Data**: Real ACY Securities prices instead of Alpha Vantage
- 🤖 **Auto-Traded**: Signals automatically executed when confidence ≥ 85%
- 📊 **Broker Tickets**: MT5 ticket numbers for each trade
- 💰 **Execution Details**: Real prices, slippage, execution times
- ⚡ **Real P&L**: Actual trading results from your broker

## Security Checklist

✅ Strong HMAC secret key (64+ random characters)  
✅ HTTPS/SSL enabled on bridge service  
✅ Firewall blocking unauthorized access  
✅ Regular secret rotation (monthly)  
✅ Monitor bridge logs for suspicious activity  
✅ Test with demo account first  

## Troubleshooting

**Bridge Won't Connect to MT5:**
- Check MT5 is logged in and showing live prices
- Verify account credentials are correct
- Try restarting MT5 terminal

**Authentication Errors:**
- Verify BRIDGE_SECRET matches on both ends
- Check system clocks are synchronized
- Ensure HTTPS is properly configured

**No Auto-Trading:**
- Set AUTO_TRADE_ENABLED=true in Replit
- Check signal confidence ≥ threshold (85%)
- Verify bridge health endpoint is responding

Your MT5 integration is now ready! This gives you professional-grade trading capabilities with real ACY Securities data and automated execution.