"""
Documentation Page
"""
import streamlit as st
import requests

st.set_page_config(page_title="Documentation", page_icon="📚", layout="wide")

st.title("📚 Documentation")

# API Documentation
st.header("🔌 API Documentation")

st.markdown("""
The Forex Signal Dashboard provides a RESTful API for external integration and automation.
All endpoints return JSON responses and use standard HTTP status codes.
""")

# Public Endpoints
st.subheader("🌍 Public Endpoints")

with st.expander("GET /api/health"):
    st.code("""
# Health Check
GET http://localhost:8000/api/health

# Response
{
  "status": "healthy",
  "timestamp": "2025-09-12T10:30:00Z",
  "version": "1.0.0"
}
    """, language="json")

with st.expander("GET /api/signals/latest"):
    st.code("""
# Get Latest Signal for All Symbols
GET http://localhost:8000/api/signals/latest

# Get Latest Signal for Specific Symbol
GET http://localhost:8000/api/signals/latest?symbol=EURUSD

# Response
{
  "id": 123,
  "symbol": "EURUSD",
  "timeframe": "M1",
  "action": "BUY",
  "price": 1.08523,
  "sl": 1.08323,
  "tp": 1.08723,
  "confidence": 0.72,
  "strategy": "ema_rsi",
  "version": "v1",
  "expires_at": "2025-09-12T11:30:00Z",
  "issued_at": "2025-09-12T10:30:00Z",
  "sent_to_whatsapp": true,
  "blocked_by_risk": false,
  "risk_reason": null
}
    """, language="json")

with st.expander("GET /api/signals/recent"):
    st.code("""
# Get Recent Signals
GET http://localhost:8000/api/signals/recent?limit=50

# Filter by Symbol
GET http://localhost:8000/api/signals/recent?symbol=GBPUSD&limit=20

# Response (Array of Signal Objects)
[
  {
    "id": 123,
    "symbol": "EURUSD",
    "action": "BUY",
    "price": 1.08523,
    // ... full signal object
  },
  // ... more signals
]
    """, language="json")

with st.expander("GET /metrics"):
    st.code("""
# Prometheus Metrics
GET http://localhost:8000/metrics

# Response (Prometheus format)
# HELP signals_generated_total Total signals generated
# TYPE signals_generated_total counter
signals_generated_total 1234

# HELP whatsapp_send_total Total WhatsApp messages sent
# TYPE whatsapp_send_total counter
whatsapp_send_total 567

# HELP whatsapp_errors_total Total WhatsApp errors
# TYPE whatsapp_errors_total counter
whatsapp_errors_total 12
    """, language="text")

# Protected Endpoints (Admin Only)
st.subheader("🔐 Protected Endpoints (Admin Only)")

st.info("These endpoints require JWT authentication. Include `Authorization: Bearer <token>` header.")

with st.expander("POST /api/auth/login"):
    st.code("""
# User Authentication
POST http://localhost:8000/api/auth/login
Content-Type: application/json

{
  "username": "admin",
  "password": "admin123"
}

# Response
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "role": "admin"
}
    """, language="json")

with st.expander("POST /api/signals/resend"):
    st.code("""
# Resend Signal to WhatsApp
POST http://localhost:8000/api/signals/resend
Authorization: Bearer <token>
Content-Type: application/json

{
  "signal_id": 123
}

# Response
{
  "status": "sent",
  "message_id": "wamid.ABC123..."
}
    """, language="json")

with st.expander("POST /api/whatsapp/test"):
    st.code("""
# Test WhatsApp Connection
POST http://localhost:8000/api/whatsapp/test
Authorization: Bearer <token>

# Response
{
  "status": "success",
  "result": {
    "total_recipients": 2,
    "successful_sends": 2,
    "failed_sends": 0
  }
}
    """, language="json")

with st.expander("POST /api/risk/killswitch"):
    st.code("""
# Toggle Kill Switch
POST http://localhost:8000/api/risk/killswitch
Authorization: Bearer <token>
Content-Type: application/json

{
  "enabled": true
}

# Response
{
  "status": "success",
  "kill_switch_enabled": true
}
    """, language="json")

with st.expander("GET /api/risk/status"):
    st.code("""
# Get Risk Management Status
GET http://localhost:8000/api/risk/status

# Response
{
  "kill_switch_enabled": false,
  "daily_loss_limit": 1000.0,
  "current_daily_loss": 150.0,
  "volatility_guard_enabled": true
}
    """, language="json")

# WhatsApp Integration
st.header("📱 WhatsApp Integration")

st.markdown("""
The system automatically sends trading signals to configured WhatsApp recipients using the WhatsApp Cloud API.
""")

st.subheader("🔧 Setup Requirements")

with st.expander("1. Meta Business Account Setup"):
    st.markdown("""
    **Create Meta for Developers App:**
    1. Go to [Meta for Developers](https://developers.facebook.com/)
    2. Click "Create App" and select "Business" type
    3. Add "WhatsApp" product to your app
    4. Complete the app setup process
    
    **Get Credentials:**
    - **Phone Number ID**: Available in WhatsApp > Getting Started
    - **Access Token**: Generate a permanent token (not temporary)
    - **Webhook URL**: Optional for production message status updates
    """)

with st.expander("2. Environment Configuration"):
    st.code("""
# Required Environment Variables
WHATSAPP_TOKEN=your_permanent_user_access_token
WHATSAPP_PHONE_ID=your_phone_number_id_from_meta
WHATSAPP_TO=+1234567890,+0987654321

# Optional - for CORS configuration
CORS_ORIGINS=http://localhost:5000,https://yourdomain.com
    """, language="bash")

with st.expander("3. Recipient Requirements"):
    st.markdown("""
    **Important**: Recipients must opt-in to receive messages from your WhatsApp Business number.
    
    **Opt-in Methods:**
    - Send a message to your WhatsApp Business number first
    - Use WhatsApp Business API opt-in flows
    - Include opt-in checkbox on your website
    
    **Phone Number Format:**
    - Use E.164 format: +[country code][phone number]
    - Examples: +1234567890, +44987654321, +9665551234
    - No spaces, dashes, or special characters
    """)

st.subheader("📨 Message Format")

st.code("""
🚨 FOREX SIGNAL

EURUSD BUY @ 1.08523 | SL 1.08323 | TP 1.08723 | conf 0.72 | ema_rsi

Time: 10:30 UTC
Expires: 11:30 UTC
""", language="text")

# Signal Strategies
st.header("📊 Trading Strategies")

st.markdown("""
The system includes three built-in trading strategies that can be configured per symbol.
""")

st.subheader("📈 EMA + RSI Strategy")

with st.expander("Strategy Details"):
    st.markdown("""
    **Logic:**
    - Buy when EMA(12) crosses above EMA(26) AND RSI(14) > 50
    - Sell when EMA(12) crosses below EMA(26) AND RSI(14) < 50
    
    **Configurable Parameters:**
    - EMA Fast Period (default: 12)
    - EMA Slow Period (default: 26)
    - RSI Period (default: 14)
    - RSI Thresholds (default: 50)
    - Minimum Confidence (default: 0.6)
    - Signal Expiry (default: 60 minutes)
    
    **Stop Loss/Take Profit:**
    - ATR-based: SL = 2x ATR, TP = 3x ATR
    - Pip-based: Configurable pip values
    """)

st.subheader("📊 Donchian + ATR Strategy")

with st.expander("Strategy Details"):
    st.markdown("""
    **Logic:**
    - Buy on breakout above Donchian(20) upper channel
    - Sell on breakout below Donchian(20) lower channel
    - Optional SuperTrend filter for trend confirmation
    
    **Configurable Parameters:**
    - Donchian Period (default: 20)
    - ATR Period (default: 14)
    - ATR Multiplier (default: 2.0)
    - SuperTrend Filter (default: enabled)
    - Minimum Confidence (default: 0.65)
    
    **Best For:**
    - Trending markets
    - Breakout trading
    - Higher timeframes
    """)

st.subheader("🔄 Mean Reversion + Bollinger Bands")

with st.expander("Strategy Details"):
    st.markdown("""
    **Logic:**
    - Buy when price bounces off lower Bollinger Band
    - Sell when price bounces off upper Bollinger Band
    - ADX filter to avoid trending markets
    
    **Configurable Parameters:**
    - Bollinger Bands Period (default: 20)
    - Standard Deviations (default: 2.0)
    - ADX Period (default: 14)
    - ADX Threshold (default: 25)
    - Z-Score Threshold (default: 2.0)
    
    **Best For:**
    - Ranging markets
    - Low ADX environments
    - Counter-trend trading
    """)

# Risk Management
st.header("🛡️ Risk Management")

st.markdown("""
The system includes comprehensive risk management features to protect against adverse market conditions.
""")

st.subheader("🚨 Risk Controls")

with st.expander("Kill Switch"):
    st.markdown("""
    **Purpose:** Emergency stop for all signal generation and delivery
    
    **When to Use:**
    - High-impact news events (NFP, FOMC, etc.)
    - System maintenance
    - Unusual market conditions
    - Emergency situations
    
    **Effect:**
    - Immediately stops signal generation
    - Blocks WhatsApp message delivery
    - Can be toggled instantly by admin users
    """)

with st.expander("Daily Loss Limit"):
    st.markdown("""
    **Purpose:** Limit maximum estimated daily loss
    
    **How it Works:**
    - Tracks estimated losses from signals
    - Blocks new signals when limit approached
    - Resets daily at midnight UTC
    
    **Configuration:**
    - Set to 1-2% of account balance
    - Default: $1000
    - Adjustable by admin users
    """)

with st.expander("Volatility Guard"):
    st.markdown("""
    **Purpose:** Avoid signals during high volatility periods
    
    **How it Works:**
    - Monitors ATR as percentage of price
    - Blocks signals when ATR exceeds threshold
    - Default threshold: 2% ATR
    
    **Benefits:**
    - Reduces false signals during news
    - Improves signal quality
    - Protects against unusual market moves
    """)

# Data Providers
st.header("📊 Data Providers")

st.subheader("🎭 Mock Data Provider (Default)")

with st.expander("Mock Data Details"):
    st.markdown("""
    **Purpose:** Provides synthetic market data for testing and demonstration
    
    **Features:**
    - Generates realistic OHLC data for EURUSD, GBPUSD, USDJPY
    - 7 days of minute-by-minute data
    - Includes trending and ranging periods
    - Automatic data generation if files missing
    
    **Data Location:**
    - `/data/mock/EURUSD.csv`
    - `/data/mock/GBPUSD.csv` 
    - `/data/mock/USDJPY.csv`
    
    **Format:**
    ```csv
    timestamp,open,high,low,close,volume
    2025-09-12 10:30:00,1.08523,1.08545,1.08510,1.08532,150
    ```
    """)

st.subheader("📈 Alpha Vantage Provider (Optional)")

with st.expander("Alpha Vantage Setup"):
    st.markdown("""
    **Purpose:** Real market data from Alpha Vantage API
    
    **Setup:**
    1. Get free API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
    2. Set environment variable: `ALPHAVANTAGE_KEY=your_api_key`
    3. System will automatically use real data when available
    
    **Limitations:**
    - Free tier: 5 API calls per minute, 500 per day
    - Limited forex pairs support
    - Premium plans available for higher limits
    
    **Fallback:**
    - Automatically falls back to mock data if API fails
    - No configuration changes needed
    """)

# Installation & Deployment
st.header("🚀 Installation & Deployment")

st.subheader("💻 Local Development")

with st.expander("Quick Start"):
    st.code("""
# 1. Clone repository
git clone <repository-url>
cd forex-signal-dashboard

# 2. Install dependencies
pip install streamlit fastapi sqlalchemy pandas numpy talib
pip install uvicorn requests python-jose python-multipart
pip install apscheduler structlog prometheus-client

# 3. Set environment variables
cp .env.example .env
# Edit .env with your WhatsApp credentials

# 4. Run the application
streamlit run app.py --server.port 5000

# The dashboard will be available at:
# Frontend: http://localhost:5000
# API: http://localhost:8000
    """, language="bash")

st.subheader("🐳 Docker Deployment")

with st.expander("Docker Setup"):
    st.code("""
# 1. Build the image
docker build -t forex-signal-dashboard .

# 2. Run with environment variables
docker run -p 5000:5000 -p 8000:8000 \\
  -e WHATSAPP_TOKEN=your_token \\
  -e WHATSAPP_PHONE_ID=your_phone_id \\
  -e WHATSAPP_TO="+1234567890" \\
  -e JWT_SECRET=your_secret_key \\
  forex-signal-dashboard

# 3. Or use docker-compose
docker-compose up -d
    """, language="bash")

st.subheader("☁️ Cloud Deployment")

with st.expander("Cloud Platforms"):
    st.markdown("""
    **Recommended Platforms:**
    
    **Streamlit Cloud:**
    - Deploy directly from GitHub
    - Free tier available
    - Built-in secrets management
    
    **Railway:**
    - One-click deployment
    - Automatic HTTPS
    - Built-in PostgreSQL
    
    **Heroku:**
    - Git-based deployment
    - Add-on marketplace
    - Dyno scaling
    
    **DigitalOcean App Platform:**
    - Container or source deployment
    - Managed databases
    - Load balancing
    """)

# Troubleshooting
st.header("🔧 Troubleshooting")

st.subheader("❓ Common Issues")

with st.expander("WhatsApp Messages Not Sending"):
    st.markdown("""
    **Check:**
    1. WHATSAPP_TOKEN is set correctly
    2. WHATSAPP_PHONE_ID is correct
    3. Recipients are in E.164 format (+1234567890)
    4. Recipients have opted in to receive messages
    5. Meta Business account is active
    
    **Test:**
    - Use "Test WhatsApp" button in Keys page
    - Check application logs for error details
    - Verify network connectivity to graph.facebook.com
    """)

with st.expander("No Signals Generated"):
    st.markdown("""
    **Check:**
    1. Kill switch is not enabled
    2. Strategies are enabled for symbols
    3. Mock data files exist or Alpha Vantage key is set
    4. Minimum confidence thresholds are reasonable
    5. Scheduler is running (check Overview page)
    
    **Debug:**
    - Check Recent Signals for blocked signals
    - Review Risk Management settings
    - Verify database connectivity
    """)

with st.expander("Database Connection Issues"):
    st.markdown("""
    **PostgreSQL Issues:**
    - Check DATABASE_URL format
    - Verify database server is running
    - Ensure user has correct permissions
    - Test connection with psql client
    
    **SQLite Issues:**
    - Check file permissions in application directory
    - Ensure disk space is available
    - Verify Python SQLite3 module is available
    """)

# API Testing
st.header("🧪 API Testing")

st.subheader("🔍 Test API Endpoints")

# Test health endpoint
if st.button("Test Health Endpoint"):
    try:
        response = requests.get("http://localhost:8000/api/health", timeout=5)
        if response.status_code == 200:
            st.success("✅ Health endpoint is working!")
            st.json(response.json())
        else:
            st.error(f"❌ Health check failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Connection failed: {e}")

# Test latest signals endpoint
if st.button("Test Latest Signals Endpoint"):
    try:
        response = requests.get("http://localhost:8000/api/signals/latest", timeout=5)
        if response.status_code == 200:
            st.success("✅ Latest signals endpoint is working!")
            st.json(response.json())
        elif response.status_code == 404:
            st.warning("⚠️ No signals found (this is normal for a new installation)")
        else:
            st.error(f"❌ Latest signals failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Connection failed: {e}")

# Test metrics endpoint
if st.button("Test Metrics Endpoint"):
    try:
        response = requests.get("http://localhost:8000/metrics", timeout=5)
        if response.status_code == 200:
            st.success("✅ Metrics endpoint is working!")
            st.text(response.text[:500] + "..." if len(response.text) > 500 else response.text)
        else:
            st.error(f"❌ Metrics failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Connection failed: {e}")

st.markdown("---")
st.caption("📚 For additional support, please check the application logs and GitHub issues.")
