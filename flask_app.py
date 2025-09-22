"""
Flask wrapper for ForexPulsePro deployment
This creates a Flask app that serves both the Streamlit frontend and FastAPI backend
"""
from flask import Flask, send_from_directory, jsonify, request, redirect
from flask_cors import CORS
import subprocess
import threading
import time
import os
import sys
import requests
from pathlib import Path

# Add backend to Python path
sys.path.append(str(Path(__file__).parent / "backend"))

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app, origins=["*"])

# Global variables to track services
backend_process = None
frontend_process = None
backend_port = 8080
frontend_port = 5001

def start_backend():
    """Start the FastAPI backend"""
    global backend_process
    try:
        # Set environment variables
        os.environ['LD_LIBRARY_PATH'] = '/usr/local/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
        
        # Start FastAPI backend
        backend_process = subprocess.Popen([
            sys.executable, '-m', 'uvicorn', 
            'backend.main:app', 
            '--host', '0.0.0.0', 
            '--port', str(backend_port),
            '--reload'
        ], cwd=os.path.dirname(__file__))
        
        print(f"Backend started on port {backend_port}")
        return True
    except Exception as e:
        print(f"Failed to start backend: {e}")
        return False

def start_frontend():
    """Start the Streamlit frontend"""
    global frontend_process
    try:
        # Set environment variables for frontend
        env = os.environ.copy()
        env['BACKEND_URL'] = f'http://localhost:{backend_port}'
        env['LD_LIBRARY_PATH'] = '/usr/local/lib:' + env.get('LD_LIBRARY_PATH', '')
        
        # Start Streamlit frontend
        frontend_process = subprocess.Popen([
            sys.executable, '-m', 'streamlit', 'run', 'app.py',
            '--server.port', str(frontend_port),
            '--server.address', '0.0.0.0',
            '--server.headless', 'true',
            '--browser.gatherUsageStats', 'false'
        ], cwd=os.path.dirname(__file__), env=env)
        
        print(f"Frontend started on port {frontend_port}")
        return True
    except Exception as e:
        print(f"Failed to start frontend: {e}")
        return False

def wait_for_service(port, timeout=30):
    """Wait for a service to be ready"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f'http://localhost:{port}', timeout=1)
            if response.status_code in [200, 404]:  # 404 is OK for some endpoints
                return True
        except:
            pass
        time.sleep(1)
    return False

# Initialize services on startup
def initialize_services():
    """Initialize both backend and frontend services"""
    print("Initializing ForexPulsePro services...")
    
    # Start backend
    if start_backend():
        print("Waiting for backend to be ready...")
        if wait_for_service(backend_port):
            print("✓ Backend is ready")
        else:
            print("✗ Backend failed to start properly")
    
    # Start frontend
    if start_frontend():
        print("Waiting for frontend to be ready...")
        if wait_for_service(frontend_port):
            print("✓ Frontend is ready")
        else:
            print("✗ Frontend failed to start properly")
    
    print("ForexPulsePro initialization complete!")

# Start services in a separate thread
threading.Thread(target=initialize_services, daemon=True).start()

@app.route('/')
def index():
    """Redirect to Streamlit frontend"""
    return redirect(f'http://localhost:{frontend_port}')

@app.route('/health')
def health():
    """Health check endpoint"""
    backend_healthy = False
    frontend_healthy = False
    
    try:
        response = requests.get(f'http://localhost:{backend_port}/api/health', timeout=2)
        backend_healthy = response.status_code == 200
    except:
        pass
    
    try:
        response = requests.get(f'http://localhost:{frontend_port}', timeout=2)
        frontend_healthy = response.status_code == 200
    except:
        pass
    
    return jsonify({
        'status': 'healthy' if backend_healthy and frontend_healthy else 'unhealthy',
        'backend': 'healthy' if backend_healthy else 'unhealthy',
        'frontend': 'healthy' if frontend_healthy else 'unhealthy',
        'backend_port': backend_port,
        'frontend_port': frontend_port
    })

@app.route('/api/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy_api(path):
    """Proxy API requests to FastAPI backend"""
    try:
        url = f'http://localhost:{backend_port}/api/{path}'
        
        if request.method == 'GET':
            response = requests.get(url, params=request.args, timeout=30)
        elif request.method == 'POST':
            response = requests.post(url, json=request.get_json(), params=request.args, timeout=30)
        elif request.method == 'PUT':
            response = requests.put(url, json=request.get_json(), params=request.args, timeout=30)
        elif request.method == 'DELETE':
            response = requests.delete(url, params=request.args, timeout=30)
        
        return response.content, response.status_code, response.headers.items()
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/metrics')
def metrics():
    """Proxy metrics endpoint"""
    try:
        response = requests.get(f'http://localhost:{backend_port}/metrics', timeout=10)
        return response.content, response.status_code, response.headers.items()
    except Exception as e:
        return f"Error fetching metrics: {str(e)}", 500

@app.route('/docs')
def docs():
    """Redirect to API documentation"""
    return redirect(f'http://localhost:{backend_port}/docs')

@app.route('/status')
def status():
    """Detailed status information"""
    return jsonify({
        'application': 'ForexPulsePro',
        'version': '1.0.0',
        'backend_port': backend_port,
        'frontend_port': frontend_port,
        'backend_process': backend_process.pid if backend_process else None,
        'frontend_process': frontend_process.pid if frontend_process else None,
        'environment': {
            'LD_LIBRARY_PATH': os.environ.get('LD_LIBRARY_PATH', 'Not set'),
            'BACKEND_URL': os.environ.get('BACKEND_URL', 'Not set')
        }
    })

if __name__ == '__main__':
    # Give services time to start
    time.sleep(5)
    
    # Run Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
