"""
Enhanced Authentication and Session Management for Production
"""
import streamlit as st
import hashlib
import jwt
import time
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import structlog

logger = structlog.get_logger(__name__)

# Secret key for JWT tokens - MUST be set in production
import os
JWT_SECRET = os.getenv("JWT_SECRET", st.secrets.get("JWT_SECRET", "forex_dashboard_secret_change_in_production"))
if JWT_SECRET == "forex_dashboard_secret_change_in_production":
    st.warning("⚠️ Using default JWT secret - SET JWT_SECRET environment variable in production!")
JWT_ALGORITHM = "HS256"
TOKEN_EXPIRY_HOURS = 24

# Demo credentials (replace with database lookup in production)
DEMO_USERS = {
    "admin": {
        "password_hash": "admin123_hashed",  # Replace with proper bcrypt hash
        "role": "admin",
        "name": "Admin User"
    },
    "viewer": {
        "password_hash": "viewer123_hashed",  # Replace with proper bcrypt hash  
        "role": "viewer",
        "name": "Viewer User"
    }
}

def hash_password(password: str) -> str:
    """Hash password using SHA-256 (use bcrypt in production)"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, stored_hash: str) -> bool:
    """Verify password against stored hash"""
    if stored_hash.endswith("_hashed"):  # Demo mode
        return password + "_hashed" == stored_hash
    return hash_password(password) == stored_hash

def create_jwt_token(username: str, role: str) -> str:
    """Create JWT token for user session"""
    payload = {
        "username": username,
        "role": role,
        "exp": datetime.utcnow() + timedelta(hours=TOKEN_EXPIRY_HOURS),
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_jwt_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify and decode JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("JWT token expired")
        return None
    except jwt.InvalidTokenError:
        logger.warning("Invalid JWT token")
        return None

def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate user credentials"""
    user = DEMO_USERS.get(username)
    if user and verify_password(password, user["password_hash"]):
        return {
            "username": username,
            "role": user["role"],
            "name": user["name"],
            "token": create_jwt_token(username, user["role"])
        }
    return None

def require_authentication(admin_only: bool = False) -> Optional[Dict[str, Any]]:
    """
    Require user authentication for protected pages
    
    Args:
        admin_only: If True, require admin role
        
    Returns:
        User info if authenticated, None otherwise
    """
    
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_info' not in st.session_state:
        st.session_state.user_info = None
    if 'auth_token' not in st.session_state:
        st.session_state.auth_token = None

    # Check existing authentication
    if st.session_state.authenticated and st.session_state.auth_token:
        # Verify token is still valid
        payload = verify_jwt_token(st.session_state.auth_token)
        if payload:
            # Check admin requirement
            if admin_only and payload.get('role') != 'admin':
                st.error("🚫 Admin privileges required to access this page")
                st.info("Please login with admin credentials")
                st.session_state.authenticated = False
                st.stop()
            return st.session_state.user_info
        else:
            # Token expired, clear session
            st.session_state.authenticated = False
            st.session_state.user_info = None
            st.session_state.auth_token = None

    # Show login form
    show_login_form(admin_only)
    
    return None

def show_login_form(admin_only: bool = False):
    """Display login form"""
    
    st.markdown("---")
    st.subheader("🔐 Authentication Required")
    
    if admin_only:
        st.warning("⚠️ **Admin access required** - Please login with admin credentials")
    else:
        st.info("🔑 Please login to access this feature")
    
    with st.form("login_form"):
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Demo Credentials")
            st.code("""
Admin Access:
Username: admin
Password: admin123

Viewer Access:
Username: viewer  
Password: viewer123
            """)
            
        with col2:
            st.markdown("### Login")
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            
            col_login, col_clear = st.columns(2)
            
            with col_login:
                login_clicked = st.form_submit_button("🔑 Login", type="primary", use_container_width=True)
                
            with col_clear:
                clear_clicked = st.form_submit_button("🗑️ Clear", use_container_width=True)
    
    if clear_clicked:
        st.session_state.authenticated = False
        st.session_state.user_info = None
        st.session_state.auth_token = None
        st.rerun()
    
    if login_clicked and username and password:
        with st.spinner("🔍 Authenticating..."):
            user_info = authenticate_user(username, password)
            
            if user_info:
                # Check admin requirement
                if admin_only and user_info['role'] != 'admin':
                    st.error("🚫 Admin privileges required for this page")
                    return
                    
                # Successful authentication
                st.session_state.authenticated = True
                st.session_state.user_info = user_info
                st.session_state.auth_token = user_info['token']
                
                st.success(f"✅ Welcome back, {user_info['name']}! ({user_info['role'].title()})")
                logger.info(f"User authenticated: {username} ({user_info['role']})")
                
                time.sleep(1)  # Brief pause for UX
                st.rerun()
            else:
                st.error("❌ Invalid credentials - Please try again")
                logger.warning(f"Failed authentication attempt for username: {username}")

    st.stop()

def logout_user():
    """Logout current user"""
    if st.session_state.get('user_info'):
        logger.info(f"User logged out: {st.session_state.user_info.get('username')}")
    
    st.session_state.authenticated = False
    st.session_state.user_info = None
    st.session_state.auth_token = None

def render_user_info():
    """Render current user info in sidebar"""
    if st.session_state.get('authenticated') and st.session_state.get('user_info'):
        user = st.session_state.user_info
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 👤 Current User")
        st.sidebar.markdown(f"**{user['name']}**")
        st.sidebar.markdown(f"Role: {user['role'].title()}")
        
        if st.sidebar.button("🚪 Logout", use_container_width=True):
            logout_user()
            st.rerun()
            
        st.sidebar.markdown("---")

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_cached_user_permissions(username: str, role: str) -> Dict[str, bool]:
    """Get cached user permissions"""
    return {
        "view_signals": True,
        "view_logs": role == "admin",
        "manage_strategies": role == "admin", 
        "manage_risk": role == "admin",
        "view_api_keys": role == "admin",
        "system_control": role == "admin"
    }