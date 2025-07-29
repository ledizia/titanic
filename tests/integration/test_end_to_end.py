import pytest
import pandas as pd
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from src.main import app, load_artifact, prediction_history
import numpy as np

class TestEndToEndAPI:
    """End-to-end tests for the Titanic API."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.client = TestClient(app)
        # Clear prediction history before each test
        prediction_history.clear()
    
    def test_health_check(self):
        """Test the health check endpoint."""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
    
    def test_metrics_endpoint(self):
        """Test the metrics endpoint."""
        response = self.client.get("/metrics")
        assert response.status_code == 200
        # Check that it returns Prometheus format
        content = response.text
        assert "titanic_prediction_requests_total" in content
    
    def test_history_endpoint(self):
        """Test the history endpoint."""
        response = self.client.get("/history")
        assert response.status_code == 200
        data = response.json()
        assert "history" in data
        assert isinstance(data["history"], list)
    
    def test_prediction_workflow(self, sample_passenger_data):
        """Test the complete prediction workflow."""
        # Ensure a model is loaded for this test
        with patch('src.main.model') as mock_model:
            mock_model.predict.return_value = [1]
            mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
            
            # Make a prediction request
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
            
            # Check history entry structure
            history_entry = history_data["history"][0]
            assert "timestamp" in history_entry
            assert "input_data" in history_entry
            assert "prediction" in history_entry
            assert "prediction_label" in history_entry
            assert "probabilities" in history_entry
    
    def test_data_validation_workflow(self, sample_passenger_data):
        """Test data validation workflow."""
        # Test with valid data (should work)
        with patch('src.main.model') as mock_model:
            mock_model.predict.return_value = [0]
            mock_model.predict_proba.return_value = [[0.7, 0.3]]
            
            response = self.client.post("/predict", json=sample_passenger_data)
            assert response.status_code == 200
        
        # Test with invalid sex
        invalid_data = sample_passenger_data.copy()
        invalid_data["Sex"] = "invalid"
        response = self.client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error
        
        # Test with invalid embarked
        invalid_data = sample_passenger_data.copy()
        invalid_data["Embarked"] = "X"
        response = self.client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error
        
        # Test with negative fare
        invalid_data = sample_passenger_data.copy()
        invalid_data["Fare"] = -10.0
        response = self.client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    def test_model_loading_workflow(self):
        """Test model loading workflow."""
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
        
        # Test loading with empty path
        response = self.client.post("/load", json={"model_path": ""})
        assert response.status_code == 422  # Validation error
    
    def test_error_handling(self):
        """Test error handling scenarios."""
        # Test with missing required fields
        incomplete_data = {
            "Pclass": 1,
            "Name": "Test",
            # Missing other required fields
        }
        response = self.client.post("/predict", json=incomplete_data)
        assert response.status_code == 422  # Validation error
        
        # Test with wrong data types
        wrong_type_data = {
            "Pclass": "not_a_number",
            "Name": "Test",
            "Sex": "male",
            "Age": 30.0,
            "SibSp": 1,
            "Parch": 0,
            "Ticket": "A123",
            "Fare": 50.0,
            "Cabin": "C85",
            "Embarked": "S"
        }
        response = self.client.post("/predict", json=wrong_type_data)
        assert response.status_code == 422  # Validation error
    
    def test_multiple_predictions(self, sample_passenger_data):
        """Test multiple predictions and history tracking."""
        # Make multiple predictions
        for i in range(3):
            data = sample_passenger_data.copy()
            data["Name"] = f"Test Passenger {i}"
            
            with patch('src.main.model') as mock_model:
                mock_model.predict.return_value = [i % 2]  # Alternate predictions
                mock_model.predict_proba.return_value = [[0.5, 0.5]]
                
                response = self.client.post("/predict", json=data)
                assert response.status_code == 200
        
        # Check that all predictions are in history
        history_response = self.client.get("/history")
        history_data = history_response.json()
        assert len(history_data["history"]) == 3
        
        # Check that each prediction has unique data
        names = [entry["input_data"]["Name"] for entry in history_data["history"]]
        assert len(set(names)) == 3
    
    def test_api_documentation(self):
        """Test that API documentation is accessible."""
        response = self.client.get("/docs")
        assert response.status_code == 200
        
        response = self.client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data 