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

# Clean, consistent CSS styling
st.markdown("""
<style>
    /* Main styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Clean title styling */
    .dashboard-title {
        font-size: 3rem;
        font-weight: 600;
        text-align: center;
        color: #2c3e50;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 3px solid #3498db;
    }
    
    /* Simple status cards */
    .status-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        text-align: center;
    }
    
    /* Clean metric styling */
    [data-testid="metric-container"] {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Simple button styling */
    .stButton > button {
        background: #3498db;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: background-color 0.2s;
    }
    
    .stButton > button:hover {
        background: #2980b9;
    }
    
    /* Section styling */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3498db;
    }
    
    /* Quick actions section */
    .quick-actions {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 2rem;
        border-radius: 10px;
        margin: 2rem 0;
        text-align: center;
    }
    
    /* Info section */
    .info-section {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1.5rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    /* Footer styling */
    .footer-text {
        text-align: center;
        color: #2c3e50;
        font-weight: 600;
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
st.markdown("### *Clean, Simple Trading Signal Management*")
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

# Quick access section
st.markdown('<div class="section-header">⚡ Quick Start</div>', unsafe_allow_html=True)

st.markdown('<div class="quick-actions">', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📈 View Live Signals", use_container_width=True):
        st.info("Navigate to 'Overview' in the sidebar to view current trading signals →")

with col2:
    if st.button("⚙️ Configure Strategies", use_container_width=True):
        st.info("Navigate to 'Strategies' in the sidebar to enable/disable trading strategies →")

with col3:
    if st.button("🛡️ Risk Settings", use_container_width=True):
        st.info("Navigate to 'Risk' in the sidebar to manage daily limits and safety controls →")
        
st.markdown('</div>', unsafe_allow_html=True)

# System Information
st.markdown('<div class="section-header">🔗 System Information</div>', unsafe_allow_html=True)

st.markdown('<div class="info-section">', unsafe_allow_html=True)
st.markdown("""
**API Endpoints Available:**
- Health Check: `GET /api/health`
- Latest Signals: `GET /api/signals/latest`
- Recent Signals: `GET /api/signals/recent`
- Risk Status: `GET /api/risk/status`
- System Metrics: `GET /metrics`

**Server Status:** Backend API running on port 8000
""")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown('<p class="footer-text">Forex Signal Dashboard v1.0 - Production Ready ✨</p>', unsafe_allow_html=True)
