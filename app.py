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

import threading
import time
import os
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

# Main dashboard

st.title("📊 Forex Signal Dashboard")
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
st.markdown("### Quick Actions")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📈 View Signals", use_container_width=True):
        st.switch_page("pages/1_overview.py")

with col2:
    if st.button("⚙️ Configure Strategies", use_container_width=True):
        st.switch_page("pages/2_strategies.py")

with col3:
    if st.button("🛡️ Risk Management", use_container_width=True):
        st.switch_page("pages/3_risk.py")

# API Information
st.markdown("### API Endpoints")
st.code("""
Health Check: GET http://localhost:8000/api/health
Latest Signals: GET http://localhost:8000/api/signals/latest
Recent Signals: GET http://localhost:8000/api/signals/recent
Metrics: GET http://localhost:8000/metrics
""")

st.markdown("---")
st.markdown("*Forex Signal Dashboard v1.0 - Production Ready*")
