import numpy as np
import joblib
import tempfile
import os
from typing import List, Optional

"""
Mock models for testing the Titanic API.
Contains mock model classes that can be serialized and used in tests.
"""

class MockModel:
    """A simple mock model that can be pickled and used in tests."""
    
    def __init__(self, predict_value: int = 1, predict_proba_value: Optional[List[float]] = None):
        """
        Initialize a mock model.
        
        Args:
            predict_value: The value to return from predict() method
            predict_proba_value: The probabilities to return from predict_proba() method
        """
        self.predict_value = predict_value
        self.predict_proba_value = predict_proba_value or [0.3, 0.7]
    
    def predict(self, X):
        """Mock predict method that returns the specified value."""
        return np.array([self.predict_value])
    
    def predict_proba(self, X):
        """Mock predict_proba method that returns the specified probabilities."""
        return np.array([self.predict_proba_value])


class MockSurvivalModel:
    """A more sophisticated mock model that simulates survival predictions."""
    
    def __init__(self, survival_rate: float = 0.4):
        """
        Initialize a mock survival model.
        
        Args:
            survival_rate: Probability of survival (0.0 to 1.0)
        """
        self.survival_rate = survival_rate
    
    def predict(self, X):
        """Predict survival based on passenger characteristics."""
        predictions = []
        for _ in range(len(X)):
            # Simple logic: higher class, female, and younger passengers have higher survival
            if hasattr(X, 'iloc'):
                # DataFrame input
                pclass = X.iloc[0]['Pclass'] if len(X) > 0 else 3
                sex_female = X.iloc[0]['Sex_female'] if len(X) > 0 else 0
                fare = X.iloc[0]['Fare'] if len(X) > 0 else 10.0
            else:
                # Array input
                pclass = X[0][0] if len(X) > 0 and len(X[0]) > 0 else 3
                sex_female = X[0][5] if len(X) > 0 and len(X[0]) > 5 else 0
                fare = X[0][4] if len(X) > 0 and len(X[0]) > 4 else 10.0
            
            # Simple survival logic
            survival_prob = 0.3  # Base probability
            if pclass == 1:
                survival_prob += 0.3
            elif pclass == 2:
                survival_prob += 0.1
            
            if sex_female > 0:  # Female
                survival_prob += 0.4
            
            if fare > 50:  # Higher fare
                survival_prob += 0.1
            
            # Add some randomness
            survival_prob += np.random.normal(0, 0.1)
            survival_prob = max(0.0, min(1.0, survival_prob))
            
            predictions.append(1 if survival_prob > 0.5 else 0)
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """Predict survival probabilities."""
        predictions = self.predict(X)
        probabilities = []
        
        for pred in predictions:
            if pred == 1:
                prob = [0.2, 0.8]  # 80% chance of survival
            else:
                prob = [0.8, 0.2]  # 80% chance of not surviving
            
            # Add some randomness
            prob = [max(0.0, min(1.0, p + np.random.normal(0, 0.05))) for p in prob]
            # Normalize
            total = sum(prob)
            prob = [p / total for p in prob]
            probabilities.append(prob)
        
        return np.array(probabilities)


def create_mock_model_file(predict_value: int = 1, predict_proba_value: Optional[List[float]] = None) -> str:
    """
    Create a temporary mock model file.
    
    Args:
        predict_value: The value to return from predict() method
        predict_proba_value: The probabilities to return from predict_proba() method
    
    Returns:
        Path to the temporary model file
    """
    mock_model = MockModel(predict_value, predict_proba_value)
    
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
        joblib.dump(mock_model, tmp_file.name)
        return tmp_file.name


def create_survival_model_file(survival_rate: float = 0.4) -> str:
    """
    Create a temporary survival model file.
    
    Args:
        survival_rate: Probability of survival (0.0 to 1.0)
    
    Returns:
        Path to the temporary model file
    """
    mock_model = MockSurvivalModel(survival_rate)
    
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
        joblib.dump(mock_model, tmp_file.name)
        return tmp_file.name


def cleanup_model_file(file_path: str) -> None:
    """
    Clean up a temporary model file.
    
    Args:
        file_path: Path to the model file to delete
    """
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(f"Warning: Could not delete temporary file {file_path}: {e}")


class MockModelContext:
    """Context manager for creating and cleaning up mock model files."""
    
    def __init__(self, predict_value: int = 1, predict_proba_value: Optional[List[float]] = None):
        self.predict_value = predict_value
        self.predict_proba_value = predict_proba_value
        self.file_path = None
    
    def __enter__(self):
        self.file_path = create_mock_model_file(self.predict_value, self.predict_proba_value)
        return self.file_path
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file_path:
            cleanup_model_file(self.file_path)


class MockSurvivalModelContext:
    """Context manager for creating and cleaning up survival model files."""
    
    def __init__(self, survival_rate: float = 0.4):
        self.survival_rate = survival_rate
        self.file_path = None
    
    def __enter__(self):
        self.file_path = create_survival_model_file(self.survival_rate)
        return self.file_path
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file_path:
            cleanup_model_file(self.file_path) 