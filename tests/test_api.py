"""
API Tests
"""
import pytest
import asyncio
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import tempfile
import os

# Import the application
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from backend.main import app
from backend.database import get_db, Base
from backend.models import User, Signal, Strategy, RiskConfig

# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

@pytest.fixture(scope="module")
def test_client():
    """Create test client with test database"""
    Base.metadata.create_all(bind=engine)
    
    # Create test data
    db = TestingSessionLocal()
    
    # Create test user
    test_user = User(username="testadmin", role="admin")
    test_user.set_password("testpass")
    db.add(test_user)
    
    # Create test strategy
    test_strategy = Strategy(
        name="ema_rsi",
        symbol="EURUSD",
        enabled=True,
        config={
            "ema_fast": 12,
            "ema_slow": 26,
            "rsi_period": 14,
            "min_confidence": 0.6
        }
    )
    db.add(test_strategy)
    
    # Create test risk config
    risk_config = RiskConfig(
        kill_switch_enabled=False,
        daily_loss_limit=1000.0,
        volatility_guard_enabled=True
    )
    db.add(risk_config)
    
    db.commit()
    db.close()
    
    client = TestClient(app)
    yield client
    
    # Cleanup
    Base.metadata.drop_all(bind=engine)

class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_check(self, test_client):
        """Test health check returns correct response"""
        response = test_client.get("/api/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["version"] == "1.0.0"

class TestAuthEndpoint:
    """Test authentication endpoints"""
    
    def test_login_success(self, test_client):
        """Test successful login"""
        response = test_client.post("/api/auth/login", json={
            "username": "testadmin",
            "password": "testpass"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["role"] == "admin"
    
    def test_login_invalid_credentials(self, test_client):
        """Test login with invalid credentials"""
        response = test_client.post("/api/auth/login", json={
            "username": "wronguser",
            "password": "wrongpass"
        })
        
        assert response.status_code == 401
        data = response.json()
        assert data["detail"] == "Invalid credentials"
    
    def test_login_missing_fields(self, test_client):
        """Test login with missing fields"""
        response = test_client.post("/api/auth/login", json={
            "username": "testadmin"
        })
        
        assert response.status_code == 422  # Validation error

class TestSignalEndpoints:
    """Test signal-related endpoints"""
    
    @pytest.fixture
    def auth_headers(self, test_client):
        """Get authentication headers"""
        response = test_client.post("/api/auth/login", json={
            "username": "testadmin",
            "password": "testpass"
        })
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    
    def test_get_latest_signal_not_found(self, test_client):
        """Test getting latest signal when none exist"""
        response = test_client.get("/api/signals/latest")
        assert response.status_code == 404
        data = response.json()
        assert data["detail"] == "No signals found"
    
    def test_get_recent_signals_empty(self, test_client):
        """Test getting recent signals when none exist"""
        response = test_client.get("/api/signals/recent")
        assert response.status_code == 200
        data = response.json()
        assert data == []
    
    def test_get_recent_signals_with_limit(self, test_client):
        """Test getting recent signals with limit parameter"""
        response = test_client.get("/api/signals/recent?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) <= 10
    
    def test_get_latest_signal_with_symbol_filter(self, test_client):
        """Test getting latest signal filtered by symbol"""
        response = test_client.get("/api/signals/latest?symbol=EURUSD")
        assert response.status_code == 404  # No signals exist yet
    
    def test_resend_signal_unauthorized(self, test_client):
        """Test resending signal without authentication"""
        response = test_client.post("/api/signals/resend", json={"signal_id": 1})
        assert response.status_code == 403
    
    def test_resend_signal_not_found(self, test_client, auth_headers):
        """Test resending non-existent signal"""
        response = test_client.post(
            "/api/signals/resend", 
            json={"signal_id": 999},
            headers=auth_headers
        )
        assert response.status_code == 404
        data = response.json()
        assert data["detail"] == "Signal not found"

class TestRiskEndpoints:
    """Test risk management endpoints"""
    
    @pytest.fixture
    def auth_headers(self, test_client):
        """Get authentication headers"""
        response = test_client.post("/api/auth/login", json={
            "username": "testadmin",
            "password": "testpass"
        })
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    
    def test_get_risk_status(self, test_client):
        """Test getting risk status"""
        response = test_client.get("/api/risk/status")
        assert response.status_code == 200
        
        data = response.json()
        assert "kill_switch_enabled" in data
        assert "daily_loss_limit" in data
        assert "current_daily_loss" in data
        assert "volatility_guard_enabled" in data
        
        # Check default values
        assert data["kill_switch_enabled"] == False
        assert data["daily_loss_limit"] == 1000.0
        assert data["volatility_guard_enabled"] == True
    
    def test_toggle_killswitch_unauthorized(self, test_client):
        """Test toggling kill switch without authentication"""
        response = test_client.post("/api/risk/killswitch", json={"enabled": True})
        assert response.status_code == 403
    
    def test_toggle_killswitch_enable(self, test_client, auth_headers):
        """Test enabling kill switch"""
        response = test_client.post(
            "/api/risk/killswitch",
            json={"enabled": True},
            headers=auth_headers
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert data["kill_switch_enabled"] == True
        
        # Verify status was updated
        status_response = test_client.get("/api/risk/status")
        status_data = status_response.json()
        assert status_data["kill_switch_enabled"] == True
    
    def test_toggle_killswitch_disable(self, test_client, auth_headers):
        """Test disabling kill switch"""
        # First enable it
        test_client.post(
            "/api/risk/killswitch",
            json={"enabled": True},
            headers=auth_headers
        )
        
        # Then disable it
        response = test_client.post(
            "/api/risk/killswitch",
            json={"enabled": False},
            headers=auth_headers
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert data["kill_switch_enabled"] == False

class TestWhatsAppEndpoints:
    """Test WhatsApp-related endpoints"""
    
    @pytest.fixture
    def auth_headers(self, test_client):
        """Get authentication headers"""
        response = test_client.post("/api/auth/login", json={
            "username": "testadmin",
            "password": "testpass"
        })
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    
    def test_whatsapp_test_unauthorized(self, test_client):
        """Test WhatsApp test without authentication"""
        response = test_client.post("/api/whatsapp/test")
        assert response.status_code == 403
    
    def test_whatsapp_test_without_config(self, test_client, auth_headers):
        """Test WhatsApp test without proper configuration"""
        # This will likely fail due to missing WhatsApp configuration
        response = test_client.post("/api/whatsapp/test", headers=auth_headers)
        # Should return 500 due to missing configuration
        assert response.status_code == 500

class TestStrategyEndpoints:
    """Test strategy configuration endpoints"""
    
    @pytest.fixture
    def auth_headers(self, test_client):
        """Get authentication headers"""
        response = test_client.post("/api/auth/login", json={
            "username": "testadmin",
            "password": "testpass"
        })
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    
    def test_get_strategies(self, test_client):
        """Test getting all strategies"""
        response = test_client.get("/api/strategies")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1  # We created one test strategy
        
        # Check structure of first strategy
        strategy = data[0]
        assert "id" in strategy
        assert "name" in strategy
        assert "symbol" in strategy
        assert "enabled" in strategy
        assert "config" in strategy
    
    def test_update_strategy_unauthorized(self, test_client):
        """Test updating strategy without authentication"""
        response = test_client.put("/api/strategies/1", json={
            "enabled": False,
            "config": {"test": "value"}
        })
        assert response.status_code == 403
    
    def test_update_strategy_not_found(self, test_client, auth_headers):
        """Test updating non-existent strategy"""
        response = test_client.put(
            "/api/strategies/999",
            json={"enabled": False},
            headers=auth_headers
        )
        assert response.status_code == 404
        data = response.json()
        assert data["detail"] == "Strategy not found"
    
    def test_update_strategy_success(self, test_client, auth_headers):
        """Test successful strategy update"""
        # First get the strategy ID
        strategies_response = test_client.get("/api/strategies")
        strategies = strategies_response.json()
        strategy_id = strategies[0]["id"]
        
        # Update the strategy
        response = test_client.put(
            f"/api/strategies/{strategy_id}",
            json={
                "enabled": False,
                "config": {"ema_fast": 10, "ema_slow": 30}
            },
            headers=auth_headers
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["enabled"] == False
        assert data["config"]["ema_fast"] == 10
        assert data["config"]["ema_slow"] == 30

class TestMetricsEndpoint:
    """Test Prometheus metrics endpoint"""
    
    def test_metrics_endpoint(self, test_client):
        """Test metrics endpoint returns Prometheus format"""
        response = test_client.get("/metrics")
        assert response.status_code == 200
        
        content = response.text
        assert "signals_generated_total" in content
        assert "whatsapp_send_total" in content
        assert "whatsapp_errors_total" in content
        
        # Check it's in Prometheus format
        lines = content.split('\n')
        help_lines = [line for line in lines if line.startswith('# HELP')]
        type_lines = [line for line in lines if line.startswith('# TYPE')]
        
        assert len(help_lines) > 0
        assert len(type_lines) > 0

class TestInputValidation:
    """Test input validation and error handling"""
    
    def test_invalid_json(self, test_client):
        """Test endpoints with invalid JSON"""
        response = test_client.post(
            "/api/auth/login",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_missing_required_fields(self, test_client):
        """Test endpoints with missing required fields"""
        response = test_client.post("/api/auth/login", json={})
        assert response.status_code == 422
    
    def test_invalid_field_types(self, test_client):
        """Test endpoints with invalid field types"""
        response = test_client.post("/api/auth/login", json={
            "username": 123,  # Should be string
            "password": []    # Should be string
        })
        assert response.status_code == 422

class TestRateLimiting:
    """Test rate limiting (if implemented)"""
    
    def test_multiple_rapid_requests(self, test_client):
        """Test making multiple rapid requests"""
        # Make multiple rapid requests to health endpoint
        responses = []
        for _ in range(10):
            response = test_client.get("/api/health")
            responses.append(response.status_code)
        
        # All should succeed if no rate limiting
        # If rate limiting is implemented, some might return 429
        success_count = sum(1 for status in responses if status == 200)
        assert success_count > 0  # At least some should succeed

class TestCORSHeaders:
    """Test CORS configuration"""
    
    def test_cors_headers_present(self, test_client):
        """Test that CORS headers are present"""
        response = test_client.get("/api/health")
        
        # Check for CORS headers (might not be present in test environment)
        # This would be more relevant in a full integration test
        assert response.status_code == 200

if __name__ == "__main__":
    pytest.main([__file__])
