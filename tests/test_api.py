"""
Tests for the FastAPI inference endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.app import app, model_state


# Create test client
client = TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_check(self):
        """Test health check returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "model_loaded" in data


class TestRootEndpoint:
    """Tests for root endpoint."""
    
    def test_root_returns_info(self):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "name" in data
        assert "version" in data


class TestPredictEndpoint:
    """Tests for prediction endpoint."""
    
    @pytest.fixture
    def sample_transaction(self):
        """Sample transaction for testing."""
        return {
            "transaction_id": "test_txn_001",
            "user_id": "test_user_123",
            "amount": 150.00,
            "timestamp": "2025-06-15T14:30:00",
            "transaction_type": "transfer",
            "merchant_category": "electronics",
            "location": "new york",
            "is_failed": False
        }
    
    def test_predict_without_model_returns_503(self, sample_transaction):
        """Test that predict returns 503 when model not loaded."""
        # Ensure model is not loaded
        original_loaded = model_state.is_loaded
        model_state.is_loaded = False
        
        try:
            response = client.post("/predict", json=sample_transaction)
            assert response.status_code == 503
        finally:
            model_state.is_loaded = original_loaded
    
    def test_predict_validates_input(self):
        """Test that predict validates input schema."""
        # Missing required field
        invalid_txn = {
            "transaction_id": "test",
            "amount": 100.0
            # Missing other required fields
        }
        
        response = client.post("/predict", json=invalid_txn)
        assert response.status_code == 422  # Validation error
    
    def test_predict_validates_amount(self, sample_transaction):
        """Test that negative amounts are rejected."""
        sample_transaction["amount"] = -100.0
        
        response = client.post("/predict", json=sample_transaction)
        assert response.status_code == 422


class TestBatchPredictEndpoint:
    """Tests for batch prediction endpoint."""
    
    @pytest.fixture
    def sample_batch(self):
        """Sample batch of transactions."""
        return {
            "transactions": [
                {
                    "transaction_id": f"test_txn_{i}",
                    "user_id": "test_user_123",
                    "amount": 100.0 + i * 10,
                    "timestamp": "2025-06-15T14:30:00",
                    "transaction_type": "payment",
                    "merchant_category": "grocery",
                    "location": "new york",
                    "is_failed": False
                }
                for i in range(3)
            ]
        }
    
    def test_batch_predict_without_model_returns_503(self, sample_batch):
        """Test that batch predict returns 503 when model not loaded."""
        original_loaded = model_state.is_loaded
        model_state.is_loaded = False
        
        try:
            response = client.post("/predict/batch", json=sample_batch)
            assert response.status_code == 503
        finally:
            model_state.is_loaded = original_loaded
    
    def test_batch_empty_list_rejected(self):
        """Test that empty batch is rejected."""
        response = client.post("/predict/batch", json={"transactions": []})
        assert response.status_code == 422


class TestModelInfoEndpoint:
    """Tests for model info endpoint."""
    
    def test_model_info_without_model_returns_503(self):
        """Test that model info returns 503 when model not loaded."""
        original_loaded = model_state.is_loaded
        model_state.is_loaded = False
        
        try:
            response = client.get("/model/info")
            assert response.status_code == 503
        finally:
            model_state.is_loaded = original_loaded


class TestThresholdEndpoint:
    """Tests for threshold update endpoint."""
    
    def test_threshold_update_validates_range(self):
        """Test that threshold must be between 0 and 1."""
        # First need model loaded
        original_loaded = model_state.is_loaded
        model_state.is_loaded = True
        
        try:
            # Invalid threshold
            response = client.post("/threshold/update?new_threshold=1.5")
            assert response.status_code == 400
            
            response = client.post("/threshold/update?new_threshold=-0.1")
            assert response.status_code == 400
        finally:
            model_state.is_loaded = original_loaded


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
