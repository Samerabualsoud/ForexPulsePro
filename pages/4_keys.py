"""
API Keys and Configuration Management Page
"""
import streamlit as st
import requests
import os
from typing import Dict, Any

st.set_page_config(page_title="API Keys", page_icon="🔐", layout="wide")

st.title("🔐 API Keys & Configuration")

# Helper function to call backend API
def call_api(endpoint, method="GET", data=None, token=None):
    """Call backend API"""
    try:
        base_url = "http://0.0.0.0:8000"
        url = f"{base_url}{endpoint}"
        
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, headers=headers, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
        return None

# Authentication state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'auth_token' not in st.session_state:
    st.session_state.auth_token = None

# Authentication required
if not st.session_state.authenticated:
    st.warning("⚠️ Admin authentication required to view/modify API keys")
    
    with st.form("login_form"):
        st.subheader("🔐 Admin Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit and username and password:
            # Try to authenticate
            auth_response = call_api("/api/auth/login", "POST", {
                "username": username,
                "password": password
            })
            
            if auth_response and auth_response.get("access_token"):
                st.session_state.authenticated = True
                st.session_state.auth_token = auth_response["access_token"]
                st.session_state.user_role = auth_response.get("role", "viewer")
                st.success("✅ Authentication successful!")
                st.rerun()
            else:
                st.error("❌ Invalid credentials")
    
    st.info("💡 Use your admin credentials to access API keys and settings")
    st.stop()

# Check if user is admin
if st.session_state.get('user_role') != 'admin':
    st.error("🚫 Admin privileges required to access API key management")
    st.stop()

def mask_key(key: str, show_chars: int = 4) -> str:
    """Mask API key showing only first/last characters"""
    if not key or len(key) < show_chars * 2:
        return "Not Set" if not key else "*" * len(key)
    return f"{key[:show_chars]}{'*' * (len(key) - show_chars * 2)}{key[-show_chars:]}"

# Main configuration interface
st.info("💡 Configure your API keys and system settings below")

# Database Configuration
st.markdown("---")
st.subheader("🗄️ Database Configuration")

col1, col2 = st.columns(2)

with col1:
    database_url = os.getenv("DATABASE_URL", "")
    
    if database_url:
        if database_url.startswith("postgresql"):
            db_type = "PostgreSQL"
            db_status = "🟢 Connected"
        elif database_url.startswith("sqlite"):
            db_type = "SQLite"
            db_status = "🟡 Fallback"
        else:
            db_type = "Unknown"
            db_status = "❓ Unknown"
    else:
        db_type = "SQLite (Default)"
        db_status = "🟡 Fallback"
    
    st.metric("Database Type", db_type)
    st.metric("Connection Status", db_status)

with col2:
    # Test database connection
    if st.button("🗄️ Test Database Connection", use_container_width=True):
        with st.spinner("Testing database connection..."):
            health_result = call_api("/api/health")
            
            if health_result:
                st.success("✅ Database connection successful!")
                st.json(health_result)
            else:
                st.error("❌ Database connection failed!")

# JWT Configuration
st.markdown("---")
st.subheader("🔑 JWT Authentication")

col1, col2 = st.columns(2)

with col1:
    jwt_secret = os.getenv("JWT_SECRET", "")
    jwt_status = "🟢 SET" if jwt_secret and jwt_secret != "your-secret-key-change-in-production" else "🔴 DEFAULT/MISSING"
    
    st.metric("JWT Secret", jwt_status)
    
    if jwt_secret == "your-secret-key-change-in-production":
        st.warning("⚠️ Using default JWT secret! Change this in production!")

with col2:
    st.metric("Token Expiry", "24 hours")
    st.metric("Algorithm", "HS256")

# Alpha Vantage Configuration (Optional)
st.markdown("---")
st.subheader("📈 Alpha Vantage API (Optional)")

col1, col2 = st.columns(2)

with col1:
    alpha_key = os.getenv("ALPHAVANTAGE_KEY", "")
    alpha_status = "🟢 SET" if alpha_key else "🔴 NOT SET"
    
    st.metric("Alpha Vantage Key", alpha_status)
    
    if alpha_key:
        st.text_input(
            "API Key",
            value=mask_key(alpha_key),
            disabled=True,
            help="Alpha Vantage API key for real market data"
        )
    else:
        st.info("💡 Alpha Vantage API key not configured. System will use mock data.")

with col2:
    st.metric("Provider Status", "Mock Data Active" if not alpha_key else "Alpha Vantage Available")
    st.caption("Mock data is generated automatically when Alpha Vantage is not configured")

# Environment Variables Guide
st.markdown("---")
st.subheader("📚 Environment Variables Guide")

with st.expander("🔧 Required Environment Variables"):
    st.markdown("""
    **WhatsApp Cloud API** (Required for notifications):
    ```bash
    WHATSAPP_TOKEN=your_permanent_user_access_token
    WHATSAPP_PHONE_ID=your_phone_number_id
    WHATSAPP_TO=+1234567890,+0987654321
    ```
    
    **Database** (Optional - defaults to SQLite):
    ```bash
    DATABASE_URL=postgresql://user:password@localhost:5432/forex_signals
    ```
    
    **Security** (Recommended to change):
    ```bash
    JWT_SECRET=your_secure_random_secret_key_here
    ```
    
    **Data Provider** (Optional - defaults to mock data):
    ```bash
    ALPHAVANTAGE_KEY=your_alpha_vantage_api_key
    ```
    """)

with st.expander("📱 How to Get WhatsApp Cloud API Credentials"):
    st.markdown("""
    **Step 1: Create Meta Business Account**
    1. Go to [Meta for Developers](https://developers.facebook.com/)
    2. Create a new app and select "Business" type
    3. Add "WhatsApp" product to your app
    
    **Step 2: Get Phone Number ID**
    1. Go to WhatsApp > Getting Started
    2. Copy the Phone Number ID from the dashboard
    3. This is your `WHATSAPP_PHONE_ID`
    
    **Step 3: Get Access Token**
    1. Generate a permanent access token (not temporary)
    2. This is your `WHATSAPP_TOKEN`
    3. Keep this token secure and private
    
    **Step 4: Add Recipients**
    1. Recipients must opt-in to receive messages
    2. Use E.164 format: +countrycode + phone number
    3. Separate multiple recipients with commas
    
    **Important Notes:**
    - Test with your own number first
    - Recipients must have opted in to receive messages
    - Use webhook URL for production message status updates
    """)

with st.expander("🗄️ Database Configuration Options"):
    st.markdown("""
    **PostgreSQL (Recommended for Production):**
    ```bash
    DATABASE_URL=postgresql://username:password@hostname:port/database_name
    ```
    
    **SQLite (Default Fallback):**
    - No configuration needed
    - Automatically creates `forex_signals.db` file
    - Suitable for development and testing
    
    **Connection Testing:**
    - Use the "Test Database Connection" button above
    - Check application logs for detailed error messages
    - Ensure database server is running and accessible
    """)

# Security warnings
st.markdown("---")
st.subheader("🚨 Security Warnings")

security_issues = []

if jwt_secret == "your-secret-key-change-in-production":
    security_issues.append("Default JWT secret is being used")

# WhatsApp integration removed - no longer checking WhatsApp tokens

if security_issues:
    st.error("🚨 Security Issues Detected:")
    for issue in security_issues:
        st.error(f"• {issue}")
else:
    st.success("✅ No security issues detected")

# Logout button
if st.button("🚪 Logout"):
    st.session_state.authenticated = False
    st.session_state.auth_token = None
    st.rerun()
