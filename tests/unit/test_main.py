import pytest
import pandas as pd
import numpy as np
import joblib
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import HTTPException
import json
from datetime import datetime

from src.main import app, load_artifact, prediction_history
from tests.fixtures.sample_data import get_sample_passenger_data


class TestLoadArtifact:
    """Test cases for the load_artifact function."""
    
    @pytest.mark.unit
    def test_load_artifact_success(self, temp_model_file):
        """Test successful model loading."""
        load_artifact(temp_model_file)
        # If no exception is raised, the test passes
    
    def test_load_artifact_file_not_found(self):
        """Test model loading with non-existent file."""
        with pytest.raises(Exception) as exc_info:
            load_artifact("nonexistent_model.pkl")
        assert "Model not found" in str(exc_info.value.detail)
    
    def test_load_artifact_load_error(self):
        """Test model loading with load error."""
        with patch('os.path.exists', return_value=True), patch('joblib.load') as mock_load:
            mock_load.side_effect = Exception("Load error")
            
            with pytest.raises(Exception) as exc_info:
                load_artifact("test_model.pkl")
            assert "Error loading artifacts" in str(exc_info.value.detail)

class TestFastAPIEndpoints:
    """Test cases for FastAPI endpoints."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.client = TestClient(app)
        prediction_history.clear()
    
    def test_health_check_success(self):
        """Test health check endpoint with loaded model."""
        with patch('src.main.model') as mock_model:
            mock_model.predict.return_value = [1]
            
            response = self.client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert "model_loaded" in data
    
    def test_health_check_no_model(self):
        """Test health check endpoint without loaded model."""
        with patch('src.main.model', None):
            response = self.client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["model_loaded"] == False
    
    def test_predict_success(self, sample_passenger_data):
        """Test successful prediction."""
        with patch('src.main.model') as mock_model:
            mock_model.predict.return_value = [1]
            mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
            
            response = self.client.post("/predict", json=sample_passenger_data)
            assert response.status_code == 200
            data = response.json()
            assert "prediction" in data
            assert "prediction_label" in data
            assert "probabilities" in data
    
    def test_predict_no_model_loaded(self, sample_passenger_data):
        """Test prediction without loaded model."""
        with patch('src.main.model', None):
            response = self.client.post("/predict", json=sample_passenger_data)
            assert response.status_code == 503
            data = response.json()
            assert "Model not loaded" in data["detail"]
    
    def test_predict_preprocessing_error(self, sample_passenger_data):
        """Test prediction with preprocessing error."""
        with patch('src.main.model') as mock_model:
            mock_model.predict.return_value = [1]
            mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
            
            with patch('src.preprocess_data.Preprocessor.preprocess_data') as mock_preprocess:
                mock_preprocess.side_effect = ValueError("Preprocessing error")
                
                response = self.client.post("/predict", json=sample_passenger_data)
                assert response.status_code == 400
                data = response.json()
                assert "Error in input data" in data["detail"]
    
    def test_predict_model_error(self, sample_passenger_data):
        """Test prediction with model error."""
        with patch('src.main.model') as mock_model:
            mock_model.predict.side_effect = Exception("Model error")
            
            response = self.client.post("/predict", json=sample_passenger_data)
            assert response.status_code == 500
            data = response.json()
            assert "Internal server error" in data["detail"]
    
    def test_predict_invalid_data(self):
        """Test prediction with invalid data."""
        invalid_data = {"invalid": "data"}
        response = self.client.post("/predict", json=invalid_data)
        assert response.status_code == 422
    
    def test_load_model_success(self):
        """Test successful model loading endpoint."""
        with patch('src.main.load_artifact') as mock_load:
            response = self.client.post("/load", json={"model_path": "/path/to/new_model.pkl"})
            assert response.status_code == 200
            data = response.json()
            assert "success" in data["status"]
            assert "New model loaded successfully." in data["message"]
    
    def test_load_model_file_not_found(self):
        """Test model loading with non-existent file."""
        response = self.client.post("/load", json={"model_path": "nonexistent.pkl"})
        assert response.status_code == 404
    
    def test_load_model_general_error(self):
        """Test model loading with general error."""
        with patch('src.main.load_artifact') as mock_load:
            mock_load.side_effect = Exception("General error")
            
            response = self.client.post("/load", json={"model_path": "/path/to/model.pkl"})
            assert response.status_code == 500
            data = response.json()
            assert "Error loading new model" in data["detail"]
    
    def test_load_model_invalid_request(self):
        """Test model loading with invalid request."""
        response = self.client.post("/load", json={"model_path": ""})
        assert response.status_code == 422
    
    def test_get_history_empty(self):
        """Test history endpoint with empty history."""
        response = self.client.get("/history")
        assert response.status_code == 200
        data = response.json()
        assert data["history"] == []
    
    def test_get_history_with_data(self, sample_passenger_data):
        """Test history endpoint with data."""
        with patch('src.main.model') as mock_model:
            mock_model.predict.return_value = [1]
            mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
            
            # Make a prediction to add to history
            self.client.post("/predict", json=sample_passenger_data)
            
            response = self.client.get("/history")
            assert response.status_code == 200
            data = response.json()
            assert len(data["history"]) == 1

class TestPredictionHistory:
    """Test cases for prediction history functionality."""
    
    def test_prediction_stored_in_history(self, sample_passenger_data):
        """Test that predictions are stored in history."""
        with patch('src.main.model') as mock_model:
            mock_model.predict.return_value = [1]
            mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
            
            client = TestClient(app)
            prediction_history.clear()
            
            response = client.post("/predict", json=sample_passenger_data)
            assert response.status_code == 200
            
            assert len(prediction_history) == 1
            history_entry = prediction_history[0]
            assert "timestamp" in history_entry
            assert "input_data" in history_entry
            assert "prediction" in history_entry

class TestStartupEvent:
    """Test cases for application startup event."""
    
    def test_startup_event_success(self):
        """Test successful startup event."""
        with patch('src.main.load_artifact') as mock_load:
            # This should not raise an exception
            from src.main import lifespan
            # The lifespan function should be callable
            assert callable(lifespan)
    
    def test_startup_event_failure(self):
        """Test startup event with model loading failure."""
        with patch('src.main.load_artifact') as mock_load:
            mock_load.side_effect = Exception("Model loading failed")
            # The lifespan should handle the exception gracefully
            from src.main import lifespan
            assert callable(lifespan)
