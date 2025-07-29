import pytest
import pandas as pd
import tempfile
import os
import joblib
import numpy as np
from unittest.mock import Mock


@pytest.fixture
def sample_titanic_data():
    """Fixture providing realistic Titanic dataset samples."""
    return pd.DataFrame({
        'Pclass': [1, 2, 3, 1, 2, 3],
        'Name': [
            'Braund, Mr. Owen Harris',
            'Cumings, Mrs. John Bradley',
            'Heikkinen, Miss. Laina',
            'Futrelle, Master. Jacques Heath',
            'Allen, Mr. William Henry',
            'Moran, Mr. James'
        ],
        'Sex': ['male', 'female', 'female', 'male', 'male', 'male'],
        'Age': [30, 25, 20, 8, 35, 27],
        'SibSp': [1, 0, 2, 3, 0, 0],
        'Parch': [0, 1, 0, 1, 0, 0],
        'Ticket': ['A123', 'B456', 'C789', 'D012', 'E345', 'F678'],
        'Fare': [50.0, 30.0, 20.0, 15.0, 40.0, 25.0],
        'Cabin': ['C85', 'D10', 'E15', 'F20', 'G25', 'H30'],
        'Embarked': ['S', 'C', 'Q', 'S', 'C', 'Q']
    })


@pytest.fixture
def mock_model_file():
    """Fixture providing a temporary mock model file."""
    mock_model = Mock()
    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
    
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
        joblib.dump(mock_model, tmp_file.name)
        yield tmp_file.name
        # Clean up
        os.unlink(tmp_file.name)


@pytest.fixture
def multiple_model_files():
    """Fixture providing multiple mock model files with different behaviors."""
    # Model 1: Predicts survival
    model1 = Mock()
    model1.predict.return_value = np.array([1])
    model1.predict_proba.return_value = np.array([[0.2, 0.8]])
    
    # Model 2: Predicts non-survival
    model2 = Mock()
    model2.predict.return_value = np.array([0])
    model2.predict_proba.return_value = np.array([[0.9, 0.1]])
    
    files = []
    
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file1:
        joblib.dump(model1, tmp_file1.name)
        files.append(tmp_file1.name)
    
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file2:
        joblib.dump(model2, tmp_file2.name)
        files.append(tmp_file2.name)
    
    yield files
    
    # Clean up
    for file_path in files:
        os.unlink(file_path)


@pytest.fixture
def sample_passenger_requests():
    """Fixture providing sample passenger data for API requests."""
    return [
        {
            "Pclass": 1,
            "Name": "Braund, Mr. Owen Harris",
            "Sex": "male",
            "Age": 30.0,
            "SibSp": 1,
            "Parch": 0,
            "Ticket": "A123",
            "Fare": 50.0,
            "Cabin": "C85",
            "Embarked": "S"
        },
        {
            "Pclass": 2,
            "Name": "Cumings, Mrs. John Bradley",
            "Sex": "female",
            "Age": 25.0,
            "SibSp": 0,
            "Parch": 1,
            "Ticket": "B456",
            "Fare": 30.0,
            "Cabin": "D10",
            "Embarked": "C"
        },
        {
            "Pclass": 3,
            "Name": "Heikkinen, Miss. Laina",
            "Sex": "female",
            "Age": 20.0,
            "SibSp": 2,
            "Parch": 0,
            "Ticket": "C789",
            "Fare": 20.0,
            "Cabin": "E15",
            "Embarked": "Q"
        }
    ]


@pytest.fixture
def invalid_passenger_requests():
    """Fixture providing invalid passenger data for testing validation."""
    return [
        {
            "data": {
                "Pclass": 4,  # Invalid Pclass
                "Name": "Braund, Mr. Owen Harris",
                "Sex": "male",
                "Age": 30.0,
                "SibSp": 1,
                "Parch": 0,
                "Ticket": "A123",
                "Fare": 50.0,
                "Cabin": "C85",
                "Embarked": "S"
            },
            "expected_status": 422
        },
        {
            "data": {
                "Pclass": 1,
                "Name": "Braund, Mr. Owen Harris",
                "Sex": "male",
                "Age": -1,  # Invalid Age
                "SibSp": 1,
                "Parch": 0,
                "Ticket": "A123",
                "Fare": 50.0,
                "Cabin": "C85",
                "Embarked": "S"
            },
            "expected_status": 422
        },
        {
            "data": {
                "Pclass": 1,
                "Name": "Braund, Mr. Owen Harris",
                "Sex": "male",
                "Age": 30.0,
                "SibSp": 1,
                "Parch": 0,
                "Ticket": "A123",
                "Fare": 50.0,
                "Cabin": "C85",
                "Embarked": "X"  # Invalid Embarked
            },
            "expected_status": 422
        }
    ]


@pytest.fixture(autouse=True)
def reset_prediction_history():
    """Fixture to reset prediction history before each integration test."""
    from src.main import prediction_history
    prediction_history.clear()
    yield
    prediction_history.clear()


@pytest.fixture(autouse=True)
def reset_model_state():
    """Fixture to reset model state before each integration test."""
    from src.main import model
    import src.main
    
    # Store original model
    original_model = src.main.model
    
    # Reset to None
    src.main.model = None
    
    yield
    
    # Restore original model
    src.main.model = original_model 