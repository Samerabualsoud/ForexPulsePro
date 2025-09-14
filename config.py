"""
Configuration utilities for Streamlit application
Handles environment-based configuration for deployment
"""
import os


def get_backend_url() -> str:
    """
    Get the backend API URL from environment variables.
    
    For deployment, this allows the backend URL to be configured
    based on the environment without hardcoding localhost addresses.
    
    Returns:
        str: The backend API base URL
    """
    # Check for explicit backend URL environment variable
    backend_url = os.getenv("BACKEND_URL")
    if backend_url:
        return backend_url.rstrip('/')
    
    # For production deployment, construct URL based on host/port
    backend_host = os.getenv("BACKEND_HOST", "localhost")
    backend_port = os.getenv("BACKEND_PORT", "8080")
    
    return f"http://{backend_host}:{backend_port}"


def get_frontend_port() -> int:
    """
    Get the frontend port from environment variables.
    
    Returns:
        int: The frontend port number
    """
    return int(os.getenv("FRONTEND_PORT", "5000"))


def get_backend_port() -> int:
    """
    Get the backend port from environment variables.
    
    Returns:
        int: The backend port number
    """
    return int(os.getenv("BACKEND_PORT", "8080"))