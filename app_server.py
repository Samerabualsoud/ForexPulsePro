"""
Simple Flask server for Digital Ocean deployment
Serves the Streamlit app and handles routing properly
"""
from flask import Flask, render_template_string, redirect, jsonify
import subprocess
import threading
import time
import os
import sys
import requests
import signal
from pathlib import Path

app = Flask(__name__)

# Global process tracking
streamlit_process = None
backend_process = None

def start_services():
    """Start both backend and frontend services"""
    global streamlit_process, backend_process
    
    try:
        # Set environment variables
        os.environ['LD_LIBRARY_PATH'] = '/usr/local/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
        os.environ['BACKEND_URL'] = 'http://localhost:8080'
        
        # Start FastAPI backend
        print("Starting FastAPI backend...")
        backend_process = subprocess.Popen([
            sys.executable, '-m', 'uvicorn', 
            'backend.main:app', 
            '--host', '0.0.0.0', 
            '--port', '8080'
        ], cwd=os.path.dirname(__file__))
        
        # Wait a moment for backend to start
        time.sleep(3)
        
        # Start Streamlit frontend
        print("Starting Streamlit frontend...")
        streamlit_process = subprocess.Popen([
            sys.executable, '-m', 'streamlit', 'run', 'app.py',
            '--server.port', '8501',
            '--server.address', '0.0.0.0',
            '--server.headless', 'true',
            '--browser.gatherUsageStats', 'false'
        ], cwd=os.path.dirname(__file__))
        
        print("Services started successfully!")
        
    except Exception as e:
        print(f"Error starting services: {e}")

# Start services in background thread
threading.Thread(target=start_services, daemon=True).start()

@app.route('/')
def index():
    """Main page with embedded Streamlit"""
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>ForexPulsePro Dashboard</title>
        <style>
            body { margin: 0; padding: 0; font-family: Arial, sans-serif; }
            .header { background: #1f2937; color: white; padding: 1rem; text-align: center; }
            .loading { text-align: center; padding: 2rem; }
            iframe { width: 100%; height: calc(100vh - 80px); border: none; }
            .error { background: #fee; border: 1px solid #fcc; padding: 1rem; margin: 1rem; border-radius: 4px; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 ForexPulsePro Dashboard</h1>
            <p>Professional Forex Signal Trading Platform</p>
        </div>
        
        <div id="content">
            <div class="loading">
                <h3>Loading Dashboard...</h3>
                <p>Please wait while the application starts up.</p>
            </div>
        </div>
        
        <script>
            // Check if Streamlit is ready and load it
            function checkStreamlit() {
                fetch('/streamlit-check')
                    .then(response => response.json())
                    .then(data => {
                        if (data.ready) {
                            document.getElementById('content').innerHTML = 
                                '<iframe src="/streamlit/" allowfullscreen></iframe>';
                        } else {
                            setTimeout(checkStreamlit, 2000);
                        }
                    })
                    .catch(error => {
                        document.getElementById('content').innerHTML = 
                            '<div class="error"><h3>Service Starting</h3><p>The dashboard is initializing. This may take up to 60 seconds.</p></div>';
                        setTimeout(checkStreamlit, 5000);
                    });
            }
            
            // Start checking after 3 seconds
            setTimeout(checkStreamlit, 3000);
        </script>
    </body>
    </html>
    ''')

@app.route('/streamlit-check')
def streamlit_check():
    """Check if Streamlit is ready"""
    try:
        response = requests.get('http://localhost:8501', timeout=2)
        return jsonify({'ready': True})
    except:
        return jsonify({'ready': False})

@app.route('/streamlit/')
@app.route('/streamlit/<path:path>')
def streamlit_proxy(path=''):
    """Proxy requests to Streamlit"""
    try:
        # Redirect to Streamlit
        return redirect(f'http://localhost:8501/{path}')
    except:
        return "Streamlit service not ready", 503

@app.route('/health')
def health():
    """Health check endpoint"""
    backend_healthy = False
    frontend_healthy = False
    
    try:
        response = requests.get('http://localhost:8080/api/health', timeout=2)
        backend_healthy = response.status_code == 200
    except:
        pass
    
    try:
        response = requests.get('http://localhost:8501', timeout=2)
        frontend_healthy = response.status_code == 200
    except:
        pass
    
    return jsonify({
        'status': 'healthy' if backend_healthy and frontend_healthy else 'starting',
        'backend': backend_healthy,
        'frontend': frontend_healthy
    })

@app.route('/api/<path:path>')
def api_proxy(path):
    """Proxy API requests to FastAPI backend"""
    try:
        response = requests.get(f'http://localhost:8080/api/{path}', timeout=10)
        return response.content, response.status_code
    except:
        return jsonify({'error': 'Backend not ready'}), 503

def cleanup():
    """Clean up processes on exit"""
    global streamlit_process, backend_process
    if streamlit_process:
        streamlit_process.terminate()
    if backend_process:
        backend_process.terminate()

# Register cleanup handler
signal.signal(signal.SIGTERM, lambda signum, frame: cleanup())

if __name__ == '__main__':
    # Give services time to start
    time.sleep(2)
    
    # Run Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
