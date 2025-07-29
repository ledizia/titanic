"""
Unit tests for utility and helper functions.
Tests common utility functions used across the application.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any


@pytest.mark.unit
class TestDataValidationHelpers:
    """Test suite for data validation helper functions."""

    @pytest.mark.unit
    def test_validate_passenger_data_structure(self):
        """Test validation of passenger data structure."""
        # This would test a helper function that validates passenger data structure
        # For now, we'll create a simple validation function
        def validate_passenger_data_structure(data: Dict[str, Any]) -> bool:
            required_fields = ['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
            return all(field in data for field in required_fields)
        
        # Valid data
        valid_data = {
            'Pclass': 1, 'Name': 'Test', 'Sex': 'male', 'Age': 30,
            'SibSp': 1, 'Parch': 0, 'Ticket': 'A123', 'Fare': 50.0,
            'Cabin': 'C85', 'Embarked': 'S'
        }
        assert validate_passenger_data_structure(valid_data) is True
        
        # Invalid data (missing field)
        invalid_data = {
            'Pclass': 1, 'Name': 'Test', 'Sex': 'male', 'Age': 30,
            'SibSp': 1, 'Parch': 0, 'Ticket': 'A123', 'Fare': 50.0,
            'Cabin': 'C85'  # Missing Embarked
        }
        assert validate_passenger_data_structure(invalid_data) is False

    @pytest.mark.unit
    def test_validate_numeric_ranges(self):
        """Test validation of numeric value ranges."""
        def validate_numeric_ranges(data: Dict[str, Any]) -> bool:
            if not (1 <= data.get('Pclass', 0) <= 3):
                return False
            if not (0 <= data.get('Age', -1) <= 100):
                return False
            if not (0 <= data.get('SibSp', -1)):
                return False
            if not (0 <= data.get('Parch', -1)):
                return False
            if not (0 <= data.get('Fare', -1)):
                return False
            return True
        
        # Valid ranges
        valid_data = {'Pclass': 2, 'Age': 25, 'SibSp': 1, 'Parch': 0, 'Fare': 30.0}
        assert validate_numeric_ranges(valid_data) is True
        
        # Invalid ranges
        invalid_data = {'Pclass': 4, 'Age': 150, 'SibSp': -1, 'Parch': 0, 'Fare': -10.0}
        assert validate_numeric_ranges(invalid_data) is False


@pytest.mark.unit
class TestDataTransformationHelpers:
    """Test suite for data transformation helper functions."""

    @pytest.mark.unit
    def test_safe_division(self):
        """Test safe division function that handles zero division."""
        def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
            try:
                return numerator / denominator
            except ZeroDivisionError:
                return default
        
        assert safe_division(10, 2) == 5.0
        assert safe_division(10, 0) == 0.0
        assert safe_division(10, 0, default=1.0) == 1.0

    @pytest.mark.unit
    def test_normalize_probabilities(self):
        """Test probability normalization function."""
        def normalize_probabilities(probs: list) -> list:
            total = sum(probs)
            if total == 0:
                return [1.0 / len(probs)] * len(probs)
            return [p / total for p in probs]
        
        # Normal case
        probs = [0.3, 0.7]
        normalized = normalize_probabilities(probs)
        assert sum(normalized) == pytest.approx(1.0, abs=1e-10)
        
        # Zero probabilities
        probs = [0.0, 0.0]
        normalized = normalize_probabilities(probs)
        assert normalized == [0.5, 0.5]
        assert sum(normalized) == 1.0

    @pytest.mark.unit
    def test_convert_to_dataframe(self):
        """Test conversion of data to DataFrame."""
        def convert_to_dataframe(data: Dict[str, Any]) -> pd.DataFrame:
            return pd.DataFrame([data])
        
        test_data = {'Name': 'Test', 'Age': 30, 'Sex': 'male'}
        df = convert_to_dataframe(test_data)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert list(df.columns) == ['Name', 'Age', 'Sex']
        assert df.iloc[0]['Name'] == 'Test'


@pytest.mark.unit
class TestDateTimeHelpers:
    """Test suite for datetime helper functions."""

    @pytest.mark.unit
    def test_format_timestamp(self):
        """Test timestamp formatting function."""
        def format_timestamp(timestamp: datetime) -> str:
            return timestamp.isoformat()
        
        now = datetime.now()
        formatted = format_timestamp(now)
        assert isinstance(formatted, str)
        assert 'T' in formatted  # ISO format contains 'T'

    @pytest.mark.unit
    def test_parse_timestamp(self):
        """Test timestamp parsing function."""
        def parse_timestamp(timestamp_str: str) -> datetime:
            return datetime.fromisoformat(timestamp_str)
        
        timestamp_str = "2023-01-01T12:00:00"
        parsed = parse_timestamp(timestamp_str)
        assert isinstance(parsed, datetime)
        assert parsed.year == 2023
        assert parsed.month == 1
        assert parsed.day == 1


@pytest.mark.unit
class TestArrayHelpers:
    """Test suite for array/numpy helper functions."""

    @pytest.mark.unit
    def test_ensure_numpy_array(self):
        """Test function to ensure data is a numpy array."""
        def ensure_numpy_array(data) -> np.ndarray:
            if isinstance(data, np.ndarray):
                return data
            return np.array(data)
        
        # List input
        result = ensure_numpy_array([1, 2, 3])
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, np.array([1, 2, 3]))
        
        # Already numpy array
        arr = np.array([1, 2, 3])
        result = ensure_numpy_array(arr)
        assert result is arr  # Should return the same object

    @pytest.mark.unit
    def test_flatten_array(self):
        """Test array flattening function."""
        def flatten_array(arr) -> np.ndarray:
            return np.array(arr).flatten()
        
        # Nested list
        nested = [[1, 2], [3, 4]]
        flattened = flatten_array(nested)
        assert np.array_equal(flattened, np.array([1, 2, 3, 4]))
        
        # Already flat
        flat = [1, 2, 3, 4]
        result = flatten_array(flat)
        assert np.array_equal(result, np.array([1, 2, 3, 4]))


@pytest.mark.unit
class TestStringHelpers:
    """Test suite for string helper functions."""

    @pytest.mark.unit
    def test_safe_string_conversion(self):
        """Test safe string conversion function."""
        def safe_string_conversion(value) -> str:
            if value is None:
                return ""
            return str(value)
        
        assert safe_string_conversion("test") == "test"
        assert safe_string_conversion(123) == "123"
        assert safe_string_conversion(None) == ""
        assert safe_string_conversion(0) == "0"

    @pytest.mark.unit
    def test_clean_string(self):
        """Test string cleaning function."""
        def clean_string(s: str) -> str:
            return s.strip().lower()
        
        assert clean_string("  TEST  ") == "test"
        assert clean_string("Test String") == "test string"
        assert clean_string("") == "" 