#!/bin/bash

# ForexPulsePro Startup Script for Digital Ocean

echo "Starting ForexPulsePro deployment..."

# Set environment variables
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/app:$PYTHONPATH

# Install TA-Lib if not present
if ! python -c "import talib" 2>/dev/null; then
    echo "Installing TA-Lib..."
    
    # Install system dependencies
    apt-get update -qq
    apt-get install -y -qq build-essential wget
    
    # Download and install TA-Lib
    cd /tmp
    wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
    tar -xzf ta-lib-0.4.0-src.tar.gz
    cd ta-lib
    ./configure --prefix=/usr/local
    make
    make install
    
    # Install Python TA-Lib
    pip install TA-Lib
    
    echo "TA-Lib installation completed"
fi

# Install any missing dependencies
pip install -r requirements.txt

# Start the application
echo "Starting ForexPulsePro application..."
cd /app
python app_server.py
