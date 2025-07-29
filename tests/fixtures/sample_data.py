"""
Sample test data for Titanic API tests.
Contains realistic passenger data from the Titanic dataset.
"""

import pandas as pd
from typing import Dict, List, Any

# Sample passenger data for unit tests
SAMPLE_PASSENGER_DATA = {
    "Pclass": 1,
    "Name": "Braund, Mr. Owen Harris",
    "Sex": "male",
    "Age": 30.0,
    "SibSp": 1,
    "Parch": 0,
    "Ticket": "A/5 21171",
    "Fare": 7.925,
    "Cabin": "C85",
    "Embarked": "S"
}

# Sample passengers for integration tests
SAMPLE_PASSENGERS = [
    {
        "Pclass": 1,
        "Name": "Braund, Mr. Owen Harris",
        "Sex": "male",
        "Age": 30.0,
        "SibSp": 1,
        "Parch": 0,
        "Ticket": "A/5 21171",
        "Fare": 7.925,
        "Cabin": "C85",
        "Embarked": "S"
    },
    {
        "Pclass": 2,
        "Name": "Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
        "Sex": "female",
        "Age": 38.0,
        "SibSp": 1,
        "Parch": 0,
        "Ticket": "PC 17599",
        "Fare": 71.2833,
        "Cabin": "C85",
        "Embarked": "C"
    },
    {
        "Pclass": 3,
        "Name": "Heikkinen, Miss. Laina",
        "Sex": "female",
        "Age": 26.0,
        "SibSp": 0,
        "Parch": 0,
        "Ticket": "STON/O2. 3101282",
        "Fare": 7.925,
        "Cabin": "",
        "Embarked": "S"
    },
    {
        "Pclass": 1,
        "Name": "Futrelle, Master. Jacques Heath",
        "Sex": "male",
        "Age": 3.0,
        "SibSp": 1,
        "Parch": 2,
        "Ticket": "113803",
        "Fare": 53.1,
        "Cabin": "C123",
        "Embarked": "S"
    },
    {
        "Pclass": 3,
        "Name": "Allen, Mr. William Henry",
        "Sex": "male",
        "Age": 35.0,
        "SibSp": 0,
        "Parch": 0,
        "Ticket": "373450",
        "Fare": 8.05,
        "Cabin": "",
        "Embarked": "S"
    }
]

# Sample passengers DataFrame for preprocessing tests
SAMPLE_PASSENGERS_DF = pd.DataFrame(SAMPLE_PASSENGERS)

# Test data for title extraction tests
TITLE_TEST_DATA = [
    {
        "name": "Braund, Mr. Owen Harris",
        "expected_title": "Mr"
    },
    {
        "name": "Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
        "expected_title": "Mrs"
    },
    {
        "name": "Heikkinen, Miss. Laina",
        "expected_title": "Miss"
    },
    {
        "name": "Futrelle, Master. Jacques Heath",
        "expected_title": "Master"
    },
    {
        "name": "Johnson, Dr. Robert",
        "expected_title": "Dr"
    },
    {
        "name": "Smith, Major John",
        "expected_title": "Major"
    },
    {
        "name": "Wilson, Mlle Sarah",
        "expected_title": "Mlle"
    },
    {
        "name": "Brown, Lady Mary",
        "expected_title": "Lady"
    },
    {
        "name": "Davis, Jonkheer Tom",
        "expected_title": "Jonkheer"
    }
]

# Test data for different passenger classes
FIRST_CLASS_PASSENGERS = [
    {
        "Pclass": 1,
        "Name": "Braund, Mr. Owen Harris",
        "Sex": "male",
        "Age": 30.0,
        "SibSp": 1,
        "Parch": 0,
        "Ticket": "A/5 21171",
        "Fare": 7.925,
        "Cabin": "C85",
        "Embarked": "S"
    },
    {
        "Pclass": 1,
        "Name": "Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
        "Sex": "female",
        "Age": 38.0,
        "SibSp": 1,
        "Parch": 0,
        "Ticket": "PC 17599",
        "Fare": 71.2833,
        "Cabin": "C85",
        "Embarked": "C"
    }
]

SECOND_CLASS_PASSENGERS = [
    {
        "Pclass": 2,
        "Name": "Heikkinen, Miss. Laina",
        "Sex": "female",
        "Age": 26.0,
        "SibSp": 0,
        "Parch": 0,
        "Ticket": "STON/O2. 3101282",
        "Fare": 7.925,
        "Cabin": "",
        "Embarked": "S"
    }
]

THIRD_CLASS_PASSENGERS = [
    {
        "Pclass": 3,
        "Name": "Allen, Mr. William Henry",
        "Sex": "male",
        "Age": 35.0,
        "SibSp": 0,
        "Parch": 0,
        "Ticket": "373450",
        "Fare": 8.05,
        "Cabin": "",
        "Embarked": "S"
    }
]

# Invalid test data for validation tests
INVALID_PASSENGER_DATA = [
    {
        "description": "Invalid Pclass",
        "data": {
            "Pclass": 4,  # Invalid: should be 1-3
            "Name": "Test, Mr. John",
            "Sex": "male",
            "Age": 30.0,
            "SibSp": 1,
            "Parch": 0,
            "Ticket": "A123",
            "Fare": 50.0,
            "Cabin": "C85",
            "Embarked": "S"
        }
    },
    {
        "description": "Invalid Age",
        "data": {
            "Pclass": 1,
            "Name": "Test, Mr. John",
            "Sex": "male",
            "Age": 150.0,  # Invalid: should be 0-100
            "SibSp": 1,
            "Parch": 0,
            "Ticket": "A123",
            "Fare": 50.0,
            "Cabin": "C85",
            "Embarked": "S"
        }
    },
    {
        "description": "Invalid Embarked",
        "data": {
            "Pclass": 1,
            "Name": "Test, Mr. John",
            "Sex": "male",
            "Age": 30.0,
            "SibSp": 1,
            "Parch": 0,
            "Ticket": "A123",
            "Fare": 50.0,
            "Cabin": "C85",
            "Embarked": "X"  # Invalid: should be C, Q, or S
        }
    }
]

def get_sample_passenger_data() -> Dict[str, Any]:
    """Get a single sample passenger data dictionary."""
    return SAMPLE_PASSENGER_DATA.copy()

def get_sample_passengers() -> List[Dict[str, Any]]:
    """Get a list of sample passenger data dictionaries."""
    return [passenger.copy() for passenger in SAMPLE_PASSENGERS]

def get_sample_passengers_df() -> pd.DataFrame:
    """Get a DataFrame with sample passenger data."""
    return SAMPLE_PASSENGERS_DF.copy()

def get_title_test_data() -> List[Dict[str, str]]:
    """Get test data for title extraction tests."""
    return TITLE_TEST_DATA.copy()

def get_invalid_passenger_data() -> List[Dict[str, Any]]:
    """Get invalid passenger data for validation tests."""
    return INVALID_PASSENGER_DATA.copy() 