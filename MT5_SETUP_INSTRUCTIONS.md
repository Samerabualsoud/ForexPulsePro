# MT5 Bridge Setup Instructions

## Overview
You now have everything set up to connect your Replit Forex Signal Dashboard to your local Windows PC with MT5 and ACY Securities account.

## Files Created
1. **`mt5_bridge_local.py`** - The bridge service to run on your Windows PC
2. **`install_mt5_bridge.bat`** - Installation script for Windows

## Step-by-Step Setup

### 1. Download Files to Your Windows PC
- Download `mt5_bridge_local.py` and `install_mt5_bridge.bat` to your Windows PC
- Place them in a folder like `C:\MT5Bridge\`

### 2. Install Required Packages
- Right-click `install_mt5_bridge.bat` and "Run as Administrator"
- This will install: MetaTrader5, fastapi, uvicorn, python-multipart

### 3. Verify MT5 is Running
- Make sure MetaTrader 5 is installed and running
- Verify you're logged in to ACY Securities account (841946)
- The terminal should show "Connected" in the bottom right

### 4. Start the Bridge Service
- Open Command Prompt as Administrator
- Navigate to your bridge folder: `cd C:\MT5Bridge\`
- Run: `python mt5_bridge_local.py`

You should see:
```
============================================================
MT5 Bridge Service - ACY Securities Integration
============================================================
Account: 841946
Server: ACYSecurities-Demo
Port: 8001
Bridge Secret: c8614671c7...
============================================================
```

### 5. Test Connection
The bridge service will be available at: `http://151.255.2.5:8001`

Test health check:
```
curl http://151.255.2.5:8001/health
```

### 6. Configure Your Replit Dashboard
Your Replit secrets are already configured:
- **MT5_BRIDGE_URL**: `http://151.255.2.5:8001`
- **MT5_BRIDGE_SECRET**: `c8614671c7d00af6bcbc578a6ce7328cc888f68b89d314a2daf4e59507c65f8b`
- **AUTO_TRADE_ENABLED**: `false` (disabled for safety)
- **AUTO_TRADE_CONFIDENCE_THRESHOLD**: `0.85`
- **AUTO_TRADE_LOT_SIZE**: `0.01`

## Security Notes
1. **Change your MT5 password** after setup - you shared it in the chat
2. The bridge secret authenticates all API calls
3. Auto-trading is disabled by default for safety

## Troubleshooting
- If the bridge can't connect to MT5, make sure MT5 is running and logged in
- Check Windows Firewall - you may need to allow port 8001
- The bridge monitors connection and auto-reconnects if MT5 disconnects
- Check `mt5_bridge.log` for detailed logs

## Next Steps
Once the bridge is running:
1. Your Replit dashboard will get real market data from ACY Securities
2. You can enable auto-trading by setting `AUTO_TRADE_ENABLED` to `true`
3. Signals with confidence ≥ 85% will be automatically executed
4. Monitor the dashboard for live trading signals and performance

Your setup is complete! The dashboard now connects to real market data through your ACY Securities MT5 account.