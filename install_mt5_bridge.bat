@echo off
echo ========================================
echo MT5 Bridge Installation Script
echo ========================================
echo.

echo Installing required Python packages...
pip install MetaTrader5 fastapi uvicorn python-multipart

echo.
echo Installation complete!
echo.
echo To run the MT5 Bridge:
echo 1. Make sure MT5 is installed and logged in to ACY Securities
echo 2. Run: python mt5_bridge_local.py
echo.
echo The bridge will run on: http://151.255.2.5:8001
echo.
pause