"""
Forex Signal Dashboard - Main Streamlit Application
"""
import streamlit as st

# Configure Streamlit page (must be first st command)
st.set_page_config(
    page_title="Forex Signal Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Custom title styling */
    .dashboard-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Status card styling */
    .status-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #e1e8ed;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        text-align: center;
    }
    
    /* Metric enhancement */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        padding: 1rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    [data-testid="metric-container"] > div {
        color: white;
    }
    
    [data-testid="metric-container"] label {
        color: rgba(255, 255, 255, 0.8);
        font-weight: 600;
    }
    
    /* Enhanced Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1rem;
    }
    
    .css-1d391kg .css-1v0mbdj {
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    /* Sidebar navigation items */
    .css-1d391kg .css-pkbazv {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        margin: 0.5rem 0;
        padding: 0.75rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .css-1d391kg .css-pkbazv:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: translateX(5px);
    }
    
    .css-1d391kg .css-pkbazv a {
        color: white !important;
        text-decoration: none;
        font-weight: 500;
    }
    
    /* Active sidebar item */
    .css-1d391kg .css-pkbazv.active {
        background: rgba(255, 255, 255, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.4);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Professional status indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-online {
        background: #10b981;
    }
    
    .status-offline {
        background: #ef4444;
    }
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Quick action section */
    .quick-actions {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        color: white;
        text-align: center;
    }
    
    /* API info section */
    .api-section {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    /* Footer styling */
    .footer-text {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: bold;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

import threading
import time
import os
import requests
import json
from pathlib import Path

# Add backend to Python path
import sys
sys.path.append(str(Path(__file__).parent / "backend"))

from backend.main import app as fastapi_app
from backend.scheduler import SignalScheduler
from backend.database import init_db, create_default_data
from backend.logs.logger import get_logger
import uvicorn

logger = get_logger(__name__)

# Initialize session state
if 'fastapi_started' not in st.session_state:
    st.session_state.fastapi_started = False
if 'scheduler_started' not in st.session_state:
    st.session_state.scheduler_started = False

def start_fastapi():
    """Start FastAPI server in background thread"""
    try:
        uvicorn.run(
            fastapi_app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"FastAPI server error: {e}")

def start_scheduler():
    """Start signal scheduler in background thread"""
    try:
        scheduler = SignalScheduler()
        scheduler.start()
        logger.info("Signal scheduler started successfully")
    except Exception as e:
        logger.error(f"Scheduler error: {e}")

def initialize_app():
    """Initialize database and background services"""
    if not st.session_state.fastapi_started:
        # Initialize database
        init_db()
        create_default_data()
        
        # Start FastAPI server
        fastapi_thread = threading.Thread(target=start_fastapi, daemon=True)
        fastapi_thread.start()
        st.session_state.fastapi_started = True
        
        # Start scheduler
        scheduler_thread = threading.Thread(target=start_scheduler, daemon=True)
        scheduler_thread.start()
        st.session_state.scheduler_started = True
        
        logger.info("Application initialized successfully")

# Initialize the application
initialize_app()

# Handle API routing through query parameters
query_params = st.query_params

# Check if this is an API request
if "api_endpoint" in query_params:
    endpoint = query_params["api_endpoint"]
    try:
        if endpoint == "health":
            response = requests.get("http://localhost:8000/api/health", timeout=5)
            st.json(response.json())
            st.stop()
        elif endpoint == "metrics":
            response = requests.get("http://localhost:8000/metrics", timeout=5)
            st.text(response.text)
            st.stop()
        elif endpoint.startswith("monitoring"):
            # Extract the full monitoring path
            monitoring_path = endpoint.replace("monitoring_", "monitoring/")
            auth_header = query_params.get("auth", "")
            headers = {"Authorization": f"Bearer {auth_header}"} if auth_header else {}
            response = requests.get(f"http://localhost:8000/api/{monitoring_path}", headers=headers, timeout=5)
            st.json(response.json())
            st.stop()
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        st.stop()

# Main dashboard
st.markdown('<h1 class="dashboard-title">📊 Forex Signal Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### *Professional Trading Signal Generation & Risk Management*")
st.markdown("---")

# Sidebar navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("Select a page from the sidebar to get started.")

# Status indicators
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "FastAPI Status",
        "Running" if st.session_state.fastapi_started else "Stopped",
        delta="Active" if st.session_state.fastapi_started else "Inactive"
    )

with col2:
    st.metric(
        "Scheduler Status", 
        "Running" if st.session_state.scheduler_started else "Stopped",
        delta="Active" if st.session_state.scheduler_started else "Inactive"
    )

with col3:
    st.metric(
        "API Port",
        "8000",
        delta="REST API available"
    )

# Quick access buttons
st.markdown('<div class="quick-actions">', unsafe_allow_html=True)
st.markdown("### 🚀 Quick Actions")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📈 View Signals", use_container_width=True):
        st.info("Navigate to 'Overview' in the sidebar to view live signals →")

with col2:
    if st.button("⚙️ Configure Strategies", use_container_width=True):
        st.info("Navigate to 'Strategies' in the sidebar to configure trading strategies →")

with col3:
    if st.button("🛡️ Risk Management", use_container_width=True):
        st.info("Navigate to 'Risk' in the sidebar to manage risk settings →")
        
st.markdown('</div>', unsafe_allow_html=True)

# API Information
st.markdown('<div class="api-section">', unsafe_allow_html=True)
st.markdown("### 🔗 API Endpoints")
st.code("""
🏥 Health Check: GET http://localhost:8000/api/health
📊 Latest Signals: GET http://localhost:8000/api/signals/latest  
📈 Recent Signals: GET http://localhost:8000/api/signals/recent
📉 Metrics: GET http://localhost:8000/metrics
🛡️ Risk Status: GET http://localhost:8000/api/risk/status
""", language="bash")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown('<p class="footer-text">Forex Signal Dashboard v1.0 - Production Ready ✨</p>', unsafe_allow_html=True)
