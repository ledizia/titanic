"""
Unit tests for Pydantic models used in the Titanic API.
Tests validation, serialization, and model behavior.
"""

import pytest
from pydantic import ValidationError
from src.main import PassengerData, LoadModelRequest

class TestPassengerData:
    """Test cases for PassengerData model validation."""
    
    def test_valid_passenger_data(self, sample_passenger_data):
        """Test that valid passenger data passes validation."""
        passenger = PassengerData(**sample_passenger_data)
        assert passenger.Pclass == 3
        assert passenger.Name == "Braund, Mr. Owen Harris"
        assert passenger.Sex == "male"
        assert passenger.Age == 22.0
        assert passenger.SibSp == 1
        assert passenger.Parch == 0
        assert passenger.Ticket == "A/5 21171"
        assert passenger.Fare == 7.25
        assert passenger.Cabin == ""
        assert passenger.Embarked == "S"

    def test_model_serialization(self, sample_passenger_data):
        """Test that the model can be serialized to dict."""
        passenger = PassengerData(**sample_passenger_data)
        try:
            # Try Pydantic v2 method first
            serialized = passenger.model_dump()
        except AttributeError:
            # Fallback to Pydantic v1 method
            serialized = passenger.dict()
        
        assert isinstance(serialized, dict)
        assert serialized["Pclass"] == 3
        assert serialized["Name"] == "Braund, Mr. Owen Harris"

    def test_model_json_serialization(self, sample_passenger_data):
        """Test that the model can be serialized to JSON."""
        passenger = PassengerData(**sample_passenger_data)
        try:
            # Try Pydantic v2 method first
            json_str = passenger.model_dump_json()
        except AttributeError:
            # Fallback to Pydantic v1 method
            json_str = passenger.json()
        
        assert isinstance(json_str, str)
        assert "Braund, Mr. Owen Harris" in json_str

    def test_invalid_sex_values(self):
        """Test that invalid sex values raise validation errors."""
        invalid_sex_data = {
            "Pclass": 3,
            "Name": "Test",
            "Sex": "invalid",
            "Age": 22.0,
            "SibSp": 1,
            "Parch": 0,
            "Ticket": "A/5 21171",
            "Fare": 7.25,
            "Cabin": "",
            "Embarked": "S"
        }
        
        with pytest.raises(ValidationError):
            PassengerData(**invalid_sex_data)

    def test_invalid_embarked_values(self):
        """Test that invalid embarked values raise validation errors."""
        invalid_embarked_data = {
            "Pclass": 3,
            "Name": "Test",
            "Sex": "male",
            "Age": 22.0,
            "SibSp": 1,
            "Parch": 0,
            "Ticket": "A/5 21171",
            "Fare": 7.25,
            "Cabin": "",
            "Embarked": "X"
        }
        
        with pytest.raises(ValidationError):
            PassengerData(**invalid_embarked_data)

    def test_invalid_fare_values(self):
        """Test that negative fare values raise validation errors."""
        invalid_fare_data = {
            "Pclass": 3,
            "Name": "Test",
            "Sex": "male",
            "Age": 22.0,
            "SibSp": 1,
            "Parch": 0,
            "Ticket": "A/5 21171",
            "Fare": -10.0,
            "Cabin": "",
            "Embarked": "S"
        }
        
        with pytest.raises(ValidationError):
            PassengerData(**invalid_fare_data)

    def test_invalid_sibsp_parch_values(self):
        """Test that negative SibSp/Parch values raise validation errors."""
        invalid_sibsp_data = {
            "Pclass": 3,
            "Name": "Test",
            "Sex": "male",
            "Age": 22.0,
            "SibSp": -1,
            "Parch": 0,
            "Ticket": "A/5 21171",
            "Fare": 7.25,
            "Cabin": "",
            "Embarked": "S"
        }
        
        with pytest.raises(ValidationError):
            PassengerData(**invalid_sibsp_data)

class TestLoadModelRequest:
    """Test cases for LoadModelRequest model validation."""
    
    def test_valid_model_path(self):
        """Test that valid model path passes validation."""
        request = LoadModelRequest(model_path="path/to/model.pkl")
        assert request.model_path == "path/to/model.pkl"

    def test_model_serialization(self):
        """Test that the model can be serialized to dict."""
        request = LoadModelRequest(model_path="path/to/model.pkl")
        try:
            # Try Pydantic v2 method first
            serialized = request.model_dump()
        except AttributeError:
            # Fallback to Pydantic v1 method
            serialized = request.dict()
        
        assert isinstance(serialized, dict)
        assert serialized["model_path"] == "path/to/model.pkl"

    def test_empty_model_path(self):
        """Test that empty model path raises validation error."""
        with pytest.raises(ValidationError):
            LoadModelRequest(model_path="")

    def test_whitespace_model_path(self):
        """Test that whitespace-only model path raises validation error."""
        with pytest.raises(ValidationError):
            LoadModelRequest(model_path="   ") 