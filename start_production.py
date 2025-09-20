#!/usr/bin/env python3
"""
Production startup script for Forex Signal Dashboard
Runs both FastAPI backend and Streamlit frontend in parallel
"""
import os
import sys
import subprocess
import time
import signal
import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionServer:
    """Manages both backend and frontend processes for production deployment"""
    
    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        self.backend_port = int(os.getenv("BACKEND_PORT", "8080"))
        self.frontend_port = int(os.getenv("FRONTEND_PORT", "5000"))
        
    def start_backend(self):
        """Start FastAPI backend server"""
        logger.info(f"Starting FastAPI backend on port {self.backend_port}")
        
        backend_cmd = [
            sys.executable, "-m", "uvicorn", 
            "backend.main:app",
            "--host", "0.0.0.0",
            "--port", str(self.backend_port),
            "--workers", "1"
        ]
        
        # Set environment variables for backend
        env = os.environ.copy()
        env.update({
            "CORS_ORIGINS": "*",
            "BACKEND_PORT": str(self.backend_port),
            "BACKEND_HOST": "0.0.0.0"
        })
        
        backend_process = subprocess.Popen(
            backend_cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        self.processes.append(backend_process)
        logger.info(f"Backend process started with PID: {backend_process.pid}")
        return backend_process
        
    def start_frontend(self):
        """Start Streamlit frontend server"""
        logger.info(f"Starting Streamlit frontend on port {self.frontend_port}")
        
        frontend_cmd = [
            "streamlit", "run", "app.py",
            "--server.port", str(self.frontend_port),
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        
        frontend_process = subprocess.Popen(
            frontend_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        self.processes.append(frontend_process)
        logger.info(f"Frontend process started with PID: {frontend_process.pid}")
        return frontend_process
        
    def wait_for_backend_ready(self, timeout=30):
        """Wait for backend to be ready"""
        import requests
        
        backend_url = f"http://localhost:{self.backend_port}/api/health"
        logger.info(f"Waiting for backend to be ready at {backend_url}")
        
        for i in range(timeout):
            try:
                response = requests.get(backend_url, timeout=2)
                if response.status_code == 200:
                    logger.info("Backend is ready!")
                    return True
            except:
                pass
            time.sleep(1)
            
        logger.warning(f"Backend not ready after {timeout} seconds")
        return False
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown()
        sys.exit(0)
        
    def shutdown(self):
        """Shutdown all processes"""
        logger.info("Shutting down all processes...")
        for process in self.processes:
            if process.poll() is None:  # Process is still running
                logger.info(f"Terminating process {process.pid}")
                process.terminate()
                
        # Wait for processes to terminate gracefully
        time.sleep(2)
        
        # Force kill if still running
        for process in self.processes:
            if process.poll() is None:
                logger.info(f"Force killing process {process.pid}")
                process.kill()
                
    def run(self):
        """Main run method"""
        logger.info("Starting Forex Signal Dashboard in production mode")
        
        # Set up signal handlers
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
        
        try:
            # Start backend first
            backend_process = self.start_backend()
            
            # Wait for backend to be ready
            if not self.wait_for_backend_ready():
                logger.error("Backend failed to start properly")
                self.shutdown()
                sys.exit(1)
                
            # Start frontend
            frontend_process = self.start_frontend()
            
            logger.info("Both services started successfully!")
            logger.info(f"Dashboard available at http://0.0.0.0:{self.frontend_port}")
            
            # Monitor processes
            while True:
                # Check if any process has died
                for process in self.processes:
                    if process.poll() is not None:
                        logger.error(f"Process {process.pid} has died unexpectedly")
                        self.shutdown()
                        sys.exit(1)
                        
                time.sleep(5)
                
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            self.shutdown()

if __name__ == "__main__":
    server = ProductionServer()
    server.run()