import pytest
import pandas as pd
import numpy as np
from src.preprocess_data import Preprocessor

class TestPreprocessor:
    """Test cases for the Preprocessor class."""
    
    def test_preprocessor_initialization(self):
        """Test that the preprocessor can be initialized."""
        preprocessor = Preprocessor()
        assert preprocessor is not None

    def test_preprocess_data_with_fixtures(self, sample_passengers_df):
        """Test preprocessing with sample data."""
        preprocessor = Preprocessor()
        processed_data = preprocessor.preprocess_data(sample_passengers_df)
        
        # Check that the output is a DataFrame
        assert isinstance(processed_data, pd.DataFrame)
        
        # Check that required features are present (Age is removed from preprocessing)
        expected_features = [
            'Pclass', 'SibSp', 'Fare', 'FamilySize', 'Alone',
            'Sex_female', 'Sex_male',
            'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Other'
        ]
        
        for feature in expected_features:
            assert feature in processed_data.columns
        
        # Check data types - should be numeric (int64, float64, or uint8)
        for col in processed_data.columns:
            assert processed_data[col].dtype in ['int64', 'float64', 'uint8']
        
        # Check that binary columns are properly encoded
        binary_columns = ['Sex_female', 'Sex_male', 'Title_Master', 'Title_Miss', 
                         'Title_Mr', 'Title_Mrs', 'Title_Other', 'Alone']
        for col in binary_columns:
            if col in processed_data.columns:
                assert processed_data[col].isin([0, 1]).all()

    def test_output_structure(self, sample_passengers_df):
        """Test that the output has the correct structure."""
        preprocessor = Preprocessor()
        processed_data = preprocessor.preprocess_data(sample_passengers_df)
        
        # Check shape - should have 3 rows and expected number of features
        assert processed_data.shape[0] == 3
        
        # Check that all expected features are present (Age is removed from preprocessing)
        expected_features = [
            'Pclass', 'SibSp', 'Fare', 'FamilySize', 'Alone',
            'Sex_female', 'Sex_male',
            'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Other'
        ]
        
        for feature in expected_features:
            assert feature in processed_data.columns
        
        # Check data types
        for col in processed_data.columns:
            assert processed_data[col].dtype in ['int64', 'float64', 'uint8']

    def test_title_extraction(self):
        """Test that titles are correctly extracted from names."""
        preprocessor = Preprocessor()
        
        # Test data with different titles
        test_data = pd.DataFrame({
            'Name': [
                'Braund, Mr. Owen Harris',
                'Cumings, Mrs. John Bradley',
                'Heikkinen, Miss. Laina',
                'Allen, Master. Hudson Trevor',
                'Dr. Smith, Dr. John',
                'Major. Jones, Major. William'
            ],
            'Sex': ['male', 'female', 'female', 'male', 'male', 'male'],
            'Age': [22, 38, 26, 4, 45, 35],
            'SibSp': [1, 1, 0, 3, 0, 0],
            'Parch': [0, 0, 0, 1, 0, 0],
            'Ticket': ['A/5 21171', 'PC 17599', 'STON/O2. 3101282', '373450', '123456', '789012'],
            'Fare': [7.25, 71.2833, 7.925, 27.9, 50.0, 30.0],
            'Cabin': ['', 'C85', '', '', '', ''],
            'Embarked': ['S', 'C', 'S', 'S', 'S', 'S'],
            'Pclass': [3, 1, 3, 3, 1, 2]
        })
        
        processed_data = preprocessor.preprocess_data(test_data)
        
        # Check that title columns are present
        title_columns = ['Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Other']
        for col in title_columns:
            assert col in processed_data.columns
        
        # Check that titles are correctly encoded
        assert processed_data['Title_Mr'].iloc[0] == 1  # Mr. Owen Harris
        assert processed_data['Title_Mrs'].iloc[1] == 1  # Mrs. John Bradley
        assert processed_data['Title_Miss'].iloc[2] == 1  # Miss. Laina
        assert processed_data['Title_Master'].iloc[3] == 1  # Master. Hudson Trevor

    def test_family_size_calculation(self):
        """Test that family size is correctly calculated."""
        preprocessor = Preprocessor()
        
        test_data = pd.DataFrame({
            'Name': ['Test, Mr. John', 'Test, Mrs. Jane'],
            'Sex': ['male', 'female'],
            'Age': [30, 25],
            'SibSp': [2, 0],
            'Parch': [1, 2],
            'Ticket': ['123', '456'],
            'Fare': [50.0, 30.0],
            'Cabin': ['', ''],
            'Embarked': ['S', 'C'],
            'Pclass': [1, 2]
        })
        
        processed_data = preprocessor.preprocess_data(test_data)
        
        # Check that Alone column is present and correctly calculated
        assert 'Alone' in processed_data.columns
        
        # First passenger: SibSp=2, Parch=1, so not alone (family_size=3)
        assert processed_data['Alone'].iloc[0] == 0
        
        # Second passenger: SibSp=0, Parch=2, so not alone (family_size=2)
        assert processed_data['Alone'].iloc[1] == 0

    def test_sex_encoding(self):
        """Test that sex is correctly encoded."""
        preprocessor = Preprocessor()
        
        test_data = pd.DataFrame({
            'Name': ['Test, Mr. John', 'Test, Mrs. Jane'],
            'Sex': ['male', 'female'],
            'Age': [30, 25],
            'SibSp': [1, 0],
            'Parch': [0, 1],
            'Ticket': ['123', '456'],
            'Fare': [50.0, 30.0],
            'Cabin': ['', ''],
            'Embarked': ['S', 'C'],
            'Pclass': [1, 2]
        })
        
        processed_data = preprocessor.preprocess_data(test_data)
        
        # Check that sex columns are present
        assert 'Sex_female' in processed_data.columns
        assert 'Sex_male' in processed_data.columns
        
        # Check encoding
        assert processed_data['Sex_male'].iloc[0] == 1  # male
        assert processed_data['Sex_female'].iloc[0] == 0  # male
        
        assert processed_data['Sex_female'].iloc[1] == 1  # female
        assert processed_data['Sex_male'].iloc[1] == 0  # female

    def test_missing_features_handling(self):
        """Test that missing features are handled correctly."""
        preprocessor = Preprocessor()
        
        # Test with minimal data that might miss some features
        test_data = pd.DataFrame({
            'Name': ['Test, Mr. John'],
            'Sex': ['male'],
            'Age': [30],
            'SibSp': [1],
            'Parch': [0],
            'Ticket': ['123'],
            'Fare': [50.0],
            'Cabin': [''],
            'Embarked': ['S'],
            'Pclass': [1]
        })
        
        processed_data = preprocessor.preprocess_data(test_data)
        
        # Check that all expected features are present (Age is removed from preprocessing)
        expected_features = [
            'Pclass', 'SibSp', 'Fare', 'FamilySize', 'Alone',
            'Sex_female', 'Sex_male',
            'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Other'
        ]
        
        for feature in expected_features:
            assert feature in processed_data.columns 