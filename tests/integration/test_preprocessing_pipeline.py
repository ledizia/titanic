import pytest
import pandas as pd
import numpy as np
from src.preprocess_data import Preprocessor

class TestPreprocessingPipeline:
    """Integration tests for the complete preprocessing pipeline."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.preprocessor = Preprocessor()
    
    def test_feature_engineering_integration(self):
        """Test complete feature engineering pipeline with real data."""
        # Create realistic test data
        test_data = pd.DataFrame({
            'Pclass': [1, 2, 3, 1, 2, 3],
            'Name': [
                'Braund, Mr. Owen Harris',
                'Cumings, Mrs. John Bradley (Florence Briggs Thayer)',
                'Heikkinen, Miss. Laina',
                'Allen, Mr. William Henry',
                'Moran, Mr. James',
                'McCarthy, Mr. Timothy J'
            ],
            'Sex': ['male', 'female', 'female', 'male', 'male', 'male'],
            'Age': [22, 38, 26, 35, 27, 54],
            'SibSp': [1, 1, 0, 0, 0, 0],
            'Parch': [0, 0, 0, 0, 0, 0],
            'Ticket': ['A/5 21171', 'PC 17599', 'STON/O2. 3101282', '373450', '330877', '17463'],
            'Fare': [7.25, 71.2833, 7.925, 8.05, 7.8292, 51.8625],
            'Cabin': ['', 'C85', '', '', '', 'E46'],
            'Embarked': ['S', 'C', 'S', 'S', 'Q', 'S']
        })
        
        # Process the data
        processed_data = self.preprocessor.preprocess_data(test_data)
        
        # Check output structure
        assert isinstance(processed_data, pd.DataFrame)
        assert len(processed_data) == 6
        
        # Check that all expected features are present (Age is removed from preprocessing)
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
        
        # Check specific feature engineering results
        # Title extraction
        assert processed_data['Title_Mr'].iloc[0] == 1  # Mr. Owen Harris
        assert processed_data['Title_Mrs'].iloc[1] == 1  # Mrs. John Bradley
        assert processed_data['Title_Miss'].iloc[2] == 1  # Miss. Laina
        
        # Sex encoding
        assert processed_data['Sex_male'].iloc[0] == 1  # male
        assert processed_data['Sex_female'].iloc[0] == 0  # male
        assert processed_data['Sex_female'].iloc[1] == 1  # female
        assert processed_data['Sex_male'].iloc[1] == 0  # female
        
        # Alone calculation
        # First passenger: SibSp=1, Parch=0, so not alone
        assert processed_data['Alone'].iloc[0] == 0
        # Third passenger: SibSp=0, Parch=0, so alone
        assert processed_data['Alone'].iloc[2] == 1
    
    def test_data_integrity_pipeline(self):
        """Test data integrity throughout the preprocessing pipeline."""
        # Create test data with various scenarios
        test_data = pd.DataFrame({
            'Pclass': [1, 2, 3, 1],
            'Name': [
                'Dr. Smith, Dr. John',
                'Major. Jones, Major. William',
                'Mlle. Wilson, Mlle. Marie',
                'Lady. Brown, Lady. Elizabeth'
            ],
            'Sex': ['male', 'male', 'female', 'female'],
            'Age': [45, 35, 25, 40],
            'SibSp': [0, 1, 0, 2],
            'Parch': [0, 1, 1, 0],
            'Ticket': ['123456', '789012', '345678', '901234'],
            'Fare': [50.0, 30.0, 25.0, 60.0],
            'Cabin': ['A1', 'B2', 'C3', 'D4'],
            'Embarked': ['S', 'C', 'Q', 'S']
        })
        
        # Store original data for comparison
        original_data = test_data.copy()
        
        # Process the data
        processed_data = self.preprocessor.preprocess_data(test_data)
        
        # Check that original data is unchanged
        pd.testing.assert_frame_equal(test_data, original_data)
        
        # Check that processed data is different
        assert not test_data.equals(processed_data)
        
        # Check that all values are numeric
        for col in processed_data.columns:
            assert processed_data[col].dtype in ['int64', 'float64', 'uint8']
        
        # Check that binary columns contain only 0 and 1
        binary_columns = ['Sex_female', 'Sex_male', 'Title_Master', 'Title_Miss', 
                         'Title_Mr', 'Title_Mrs', 'Title_Other', 'Alone']
        for col in binary_columns:
            if col in processed_data.columns:
                assert processed_data[col].isin([0, 1]).all()
    
    def test_edge_cases_pipeline(self):
        """Test preprocessing pipeline with edge cases."""
        # Test with minimal data
        minimal_data = pd.DataFrame({
            'Pclass': [1],
            'Name': ['Test, Mr. John'],
            'Sex': ['male'],
            'Age': [30],
            'SibSp': [0],
            'Parch': [0],
            'Ticket': ['123'],
            'Fare': [50.0],
            'Cabin': [''],
            'Embarked': ['S']
        })
        
        processed_data = self.preprocessor.preprocess_data(minimal_data)
        assert len(processed_data) == 1
        
        # Check that all expected features are present (Age is removed from preprocessing)
        expected_features = [
            'Pclass', 'SibSp', 'Fare', 'FamilySize', 'Alone',
            'Sex_female', 'Sex_male',
            'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Other'
        ]
        
        for feature in expected_features:
            assert feature in processed_data.columns
        
        # Test with multiple passengers having same characteristics
        duplicate_data = pd.DataFrame({
            'Pclass': [1, 1, 1],
            'Name': ['Test, Mr. John', 'Test, Mr. John', 'Test, Mr. John'],
            'Sex': ['male', 'male', 'male'],
            'Age': [30, 30, 30],
            'SibSp': [0, 0, 0],
            'Parch': [0, 0, 0],
            'Ticket': ['123', '123', '123'],
            'Fare': [50.0, 50.0, 50.0],
            'Cabin': ['', '', ''],
            'Embarked': ['S', 'S', 'S']
        })
        
        processed_duplicates = self.preprocessor.preprocess_data(duplicate_data)
        assert len(processed_duplicates) == 3
        
        # All should have same processed features
        for col in processed_duplicates.columns:
            assert processed_duplicates[col].nunique() == 1
    
    def test_title_mapping_pipeline(self):
        """Test title mapping in the preprocessing pipeline."""
        # Test various title mappings
        title_test_data = pd.DataFrame({
            'Pclass': [1, 1, 1, 1, 1, 1, 1, 1],
            'Name': [
                'Dr. Smith, Dr. John',      # Should map to Mr
                'Major. Jones, Major. William', # Should map to Mr
                'Mlle. Wilson, Mlle. Marie',    # Should map to Miss
                'Lady. Brown, Lady. Elizabeth', # Should map to Mrs
                'Jonkheer. Davis, Jonkheer. Tom', # Should map to Other
                'Capt. Miller, Capt. Robert',   # Should map to Mr
                'Sir. Taylor, Sir. William',    # Should map to Mr
                'Don. Anderson, Don. Carlos'     # Should map to Mr (not Other)
            ],
            'Sex': ['male', 'male', 'female', 'female', 'male', 'male', 'male', 'male'],
            'Age': [45, 35, 25, 40, 50, 30, 55, 45],
            'SibSp': [0, 1, 0, 2, 0, 1, 0, 1],
            'Parch': [0, 1, 1, 0, 0, 0, 1, 0],
            'Ticket': ['123', '456', '789', '012', '345', '678', '901', '234'],
            'Fare': [50.0, 30.0, 25.0, 60.0, 40.0, 35.0, 70.0, 45.0],
            'Cabin': ['A1', 'B2', 'C3', 'D4', 'E5', 'F6', 'G7', 'H8'],
            'Embarked': ['S', 'C', 'Q', 'S', 'C', 'Q', 'S', 'C']
        })
        
        processed_data = self.preprocessor.preprocess_data(title_test_data)
        
        # Check title mappings
        # Dr, Major, Capt, Sir, Don should map to Mr
        assert processed_data['Title_Mr'].iloc[0] == 1  # Dr
        assert processed_data['Title_Mr'].iloc[1] == 1  # Major
        assert processed_data['Title_Mr'].iloc[5] == 1  # Capt
        assert processed_data['Title_Mr'].iloc[6] == 1  # Sir
        assert processed_data['Title_Mr'].iloc[7] == 1  # Don (maps to Mr, not Other)
        
        # Mlle should map to Miss
        assert processed_data['Title_Miss'].iloc[2] == 1  # Mlle
        
        # Lady should map to Mrs
        assert processed_data['Title_Mrs'].iloc[3] == 1  # Lady
        
        # Jonkheer should map to Other
        assert processed_data['Title_Other'].iloc[4] == 1  # Jonkheer
    
    def test_family_calculations_pipeline(self):
        """Test family size and alone calculations in the pipeline."""
        # Test various family configurations
        family_test_data = pd.DataFrame({
            'Pclass': [1, 2, 3, 1, 2, 3],
            'Name': ['Test1', 'Test2', 'Test3', 'Test4', 'Test5', 'Test6'],
            'Sex': ['male', 'female', 'male', 'female', 'male', 'female'],
            'Age': [30, 25, 20, 35, 40, 45],
            'SibSp': [0, 1, 2, 3, 0, 1],
            'Parch': [0, 0, 0, 0, 1, 2],
            'Ticket': ['123', '456', '789', '012', '345', '678'],
            'Fare': [50.0, 30.0, 20.0, 40.0, 60.0, 35.0],
            'Cabin': ['A1', 'B2', 'C3', 'D4', 'E5', 'F6'],
            'Embarked': ['S', 'C', 'Q', 'S', 'C', 'Q']
        })
        
        processed_data = self.preprocessor.preprocess_data(family_test_data)
        
        # Check alone calculations
        # Passenger 1: SibSp=0, Parch=0 -> Alone=1
        assert processed_data['Alone'].iloc[0] == 1
        
        # Passenger 2: SibSp=1, Parch=0 -> Alone=0
        assert processed_data['Alone'].iloc[1] == 0
        
        # Passenger 3: SibSp=2, Parch=0 -> Alone=0
        assert processed_data['Alone'].iloc[2] == 0
        
        # Passenger 4: SibSp=3, Parch=0 -> Alone=0
        assert processed_data['Alone'].iloc[3] == 0
        
        # Passenger 5: SibSp=0, Parch=1 -> Alone=0
        assert processed_data['Alone'].iloc[4] == 0
        
        # Passenger 6: SibSp=1, Parch=2 -> Alone=0
        assert processed_data['Alone'].iloc[5] == 0 