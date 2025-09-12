"""
API Proxy for Streamlit to forward requests to FastAPI backend
"""
import streamlit as st
import requests
import json
from typing import Dict, Any

class APIProxy:
    """Proxy API requests from Streamlit to FastAPI backend"""
    
    def __init__(self, fastapi_base_url: str = "http://localhost:8000"):
        self.fastapi_base_url = fastapi_base_url
    
    def forward_request(self, path: str, method: str = "GET", 
                       headers: Dict[str, str] = None, 
                       data: Any = None) -> requests.Response:
        """Forward request to FastAPI backend"""
        url = f"{self.fastapi_base_url}{path}"
        
        if headers is None:
            headers = {}
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=headers, timeout=30)
            elif method.upper() == "POST":
                response = requests.post(url, headers=headers, json=data, timeout=30)
            elif method.upper() == "PUT":
                response = requests.put(url, headers=headers, json=data, timeout=30)
            elif method.upper() == "DELETE":
                response = requests.delete(url, headers=headers, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            return response
            
        except requests.RequestException as e:
            # Create a mock response for failed requests
            mock_response = requests.Response()
            mock_response.status_code = 503
            mock_response._content = json.dumps({
                "detail": f"Backend service unavailable: {str(e)}"
            }).encode()
            return mock_response

# Global proxy instance
api_proxy = APIProxy()

def handle_api_request(path: str) -> Dict[str, Any]:
    """Handle API request and return JSON response"""
    try:
        response = api_proxy.forward_request(path)
        
        if response.headers.get('content-type', '').startswith('application/json'):
            return {
                "status_code": response.status_code,
                "data": response.json(),
                "headers": dict(response.headers)
            }
        else:
            return {
                "status_code": response.status_code,
                "data": response.text,
                "headers": dict(response.headers)
            }
    except Exception as e:
        return {
            "status_code": 500,
            "data": {"detail": f"Proxy error: {str(e)}"},
            "headers": {"content-type": "application/json"}
        }

def handle_metrics_request() -> str:
    """Handle Prometheus metrics request"""
    try:
        response = api_proxy.forward_request("/metrics")
        return response.text
    except Exception as e:
        return f"# Metrics unavailable: {str(e)}\n"