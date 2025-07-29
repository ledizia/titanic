import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from src.main import app, load_artifact, prediction_history

class TestAPIIntegration:
    """Integration tests for API workflows."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.client = TestClient(app)
        prediction_history.clear()
    
    def test_complete_prediction_workflow(self, sample_passenger_data):
        """Test complete prediction workflow from request to response."""
        with patch('src.main.model') as mock_model:
            mock_model.predict.return_value = [1]
            mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
            
            # Make prediction request
            response = self.client.post("/predict", json=sample_passenger_data)
            assert response.status_code == 200
            
            data = response.json()
            assert "prediction" in data
            assert "prediction_label" in data
            assert "probabilities" in data
            assert "message" in data
            
            # Check that prediction was stored in history
            history_response = self.client.get("/history")
            history_data = history_response.json()
            assert len(history_data["history"]) == 1
    
    def test_multiple_predictions_with_history(self, sample_passenger_data):
        """Test multiple predictions and history tracking."""
        with patch('src.main.model') as mock_model:
            mock_model.predict.return_value = [0]
            mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])
            
            # Make multiple predictions
            for i in range(3):
                data = sample_passenger_data.copy()
                data["Name"] = f"Test Passenger {i}"
                
                response = self.client.post("/predict", json=data)
                assert response.status_code == 200
            
            # Check history
            history_response = self.client.get("/history")
            history_data = history_response.json()
            assert len(history_data["history"]) == 3
    
    def test_model_reloading_integration(self):
        """Test model reloading workflow."""
        # Test loading a new model
        model_path = "src/models/titanic_model.pkl"
        response = self.client.post("/load", json={"model_path": model_path})
        
        # Should work if model file exists, otherwise 404
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            assert "message" in data
        else:
            assert response.status_code == 404
        
        # Test loading with invalid path
        response = self.client.post("/load", json={"model_path": "nonexistent/path.pkl"})
        assert response.status_code == 404
    
    def test_real_preprocessing_integration(self, sample_passenger_data):
        """Test integration with real preprocessing."""
        with patch('src.main.model') as mock_model:
            mock_model.predict.return_value = [1]
            mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
            
            response = self.client.post("/predict", json=sample_passenger_data)
            assert response.status_code == 200
            
            # Check that preprocessing worked correctly
            data = response.json()
            assert "prediction" in data
            assert "probabilities" in data
    
    def test_error_handling_integration(self, sample_passenger_data):
        """Test error handling in integration scenarios."""
        # Test prediction without loaded model
        with patch('src.main.model', None):
            response = self.client.post("/predict", json=sample_passenger_data)
            assert response.status_code == 503
            data = response.json()
            assert "Model not loaded" in data["detail"]
        
        # Test with invalid data
        invalid_data = sample_passenger_data.copy()
        invalid_data["Sex"] = "invalid"
        response = self.client.post("/predict", json=invalid_data)
        assert response.status_code == 422
    
    def test_health_check_integration(self):
        """Test health check integration."""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data 