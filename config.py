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
    
    # For development/testing, try localhost first (works in Replit workspace)
    backend_host = os.getenv("BACKEND_HOST", "localhost")
    backend_port = os.getenv("BACKEND_PORT", "8080")
    localhost_url = f"http://{backend_host}:{backend_port}"
    
    # In both development and production, try localhost:8080 first
    # This works in Replit workspace and published apps where both services run together
    import requests
    try:
        # Quick health check to see if backend is accessible
        response = requests.get(f"{localhost_url}/api/health", timeout=2)
        if response.status_code == 200:
            return localhost_url
    except:
        pass
    
    # Fallback: In Replit environment, both frontend and backend run in same workspace  
    # so we can use localhost for internal communication
    replit_domain = os.getenv("REPL_SLUG") or os.getenv("REPL_ID")
    if replit_domain:
        # In Replit workspace, services communicate internally via localhost
        return localhost_url
    
    # Final fallback: return localhost (will trigger demo mode if unreachable)
    return localhost_url


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