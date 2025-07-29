import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import pandas as pd
import tempfile
import numpy as np
from unittest.mock import Mock

# Common test data
@pytest.fixture
def sample_passenger_data():
    """Sample passenger data for testing"""
    return {
        "Pclass": 3,
        "Name": "Braund, Mr. Owen Harris",
        "Sex": "male",
        "Age": 22.0,
        "SibSp": 1,
        "Parch": 0,
        "Ticket": "A/5 21171",
        "Fare": 7.25,
        "Cabin": "",
        "Embarked": "S"
    }

@pytest.fixture
def sample_passengers_df():
    """Sample passenger data as DataFrame"""
    return pd.DataFrame({
        'Pclass': [3, 1, 3],
        'Name': ['Braund, Mr. Owen Harris', 'Cumings, Mrs. John Bradley (Florence Briggs Thayer)', 'Heikkinen, Miss. Laina'],
        'Sex': ['male', 'female', 'female'],
        'Age': [22, 38, 26],
        'SibSp': [1, 1, 0],
        'Parch': [0, 0, 0],
        'Ticket': ['A/5 21171', 'PC 17599', 'STON/O2. 3101282'],
        'Fare': [7.25, 71.2833, 7.925],
        'Cabin': ['', 'C85', ''],
        'Embarked': ['S', 'C', 'S']
    })

@pytest.fixture
def mock_model():
    """Mock ML model for testing"""
    model = Mock()
    model.predict.return_value = np.array([0])  # Not survived
    model.predict_proba.return_value = np.array([[0.7, 0.3]])  # [not_survived, survived]
    return model

@pytest.fixture
def mock_preprocessor():
    """Mock preprocessor for testing"""
    preprocessor = Mock()
    preprocessor.preprocess_data.return_value = pd.DataFrame({
        'Pclass': [3],
        'Age': [22.0],
        'SibSp': [1],
        'Parch': [0],
        'Fare': [7.25],
        'Sex_female': [0],
        'Sex_male': [1],
        'Title_Master': [0],
        'Title_Miss': [0],
        'Title_Mr': [1],
        'Title_Mrs': [0],
        'Title_Other': [0],
        'Alone': [0]
    })
    return preprocessor

@pytest.fixture
def temp_model_file():
    """Fixture providing a temporary model file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
        # Create a simple model using a dictionary that can be pickled
        import joblib
        # Use a simple dictionary with numpy arrays instead of lambda functions
        simple_model = {
            'predict': np.array([0]),
            'predict_proba': np.array([[0.7, 0.3]])
        }
        joblib.dump(simple_model, tmp_file.name)
        yield tmp_file.name
        # Clean up
        os.unlink(tmp_file.name)

@pytest.fixture(autouse=True)
def reset_prediction_history():
    """Fixture to reset prediction history before each test."""
    from src.main import prediction_history
    prediction_history.clear()
    yield
    prediction_history.clear()

@pytest.fixture
def api_client():
    """Fixture providing a test client for the FastAPI app."""
    from fastapi.testclient import TestClient
    from src.main import app
    return TestClient(app) 