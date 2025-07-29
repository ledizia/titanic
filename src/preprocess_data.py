import pandas as pd
import numpy as np
import re

class Preprocessor:
    """
    Preprocessor class for Titanic dataset.
    Handles feature engineering and data preprocessing.
    """
    
    def __init__(self):
        """Initialize the preprocessor."""
        pass
    
    def extract_title(self, name):
        """
        Extract title from passenger name.
        
        Args:
            name (str): Passenger name
            
        Returns:
            str: Extracted title
        """
        title_match = re.search(r',\s*([^.]*)\.', name)
        if title_match:
            return title_match.group(1).strip()
        return "Unknown"
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the Titanic dataset.
        
        Args:
            data (pd.DataFrame): Raw passenger data
            
        Returns:
            pd.DataFrame: Preprocessed data with engineered features
        """
        processed_data = data.copy()
        
        # Extract titles from names
        processed_data['Title'] = processed_data['Name'].apply(self.extract_title)
        
        # Map titles to categories
        processed_data['Title'] = processed_data['Title'].replace(['Dr', 'Major', 'Don', 'Sir', 'Capt'], 'Mr')
        processed_data['Title'] = processed_data['Title'].replace(['Mlle', 'Mme', 'Ms'], 'Miss')
        processed_data['Title'] = processed_data['Title'].replace(['Lady', 'Countess'], 'Mrs')
        processed_data['Title'] = processed_data['Title'].replace(['Jonkheer', 'Col', 'Rev'], 'Other')
        
        # Calculate family size and alone feature
        processed_data['FamilySize'] = processed_data['SibSp'] + processed_data['Parch'] + 1
        processed_data['Alone'] = (processed_data['FamilySize'] == 1).astype(int)
        
        # Create one-hot encoding for titles and sex
        title_dummies = pd.get_dummies(processed_data['Title'], prefix='Title')
        sex_dummies = pd.get_dummies(processed_data['Sex'], prefix='Sex')
        processed_data = pd.concat([processed_data, title_dummies, sex_dummies], axis=1)

        expected_features = ['Pclass', 'SibSp', 'FamilySize', 'Alone', 'Fare', 'Sex_female', 'Sex_male', 'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Other']

        for feature in expected_features:
            if feature not in processed_data.columns:
                processed_data[feature] = 0
        
        return processed_data[expected_features]