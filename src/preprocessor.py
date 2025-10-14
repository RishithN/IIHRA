import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataPreprocessor:
    def __init__(self, users_df, trends_df, ingredient_risk_df):
        self.users_df = users_df
        self.trends_df = trends_df
        self.ingredient_risk_df = ingredient_risk_df
        self.processed_users = None
        self.processed_trends = None
        self.combined_features = None

    def preprocess_users(self):
        """Preprocess user data"""
        users_processed = self.users_df.copy()
        
        # ROBUSTNESS FIX: Fill missing categorical data before encoding/mapping
        users_processed['SkinType'].fillna('Unknown', inplace=True)
        users_processed['HormoneLevel'].fillna('Normal', inplace=True)
        
        # Handle Allergies - convert to list and create allergy columns
        users_processed['Allergies'] = users_processed['Allergies'].apply(
            lambda x: [] if str(x).lower().strip() in ('none', 'nan') 
              else [allergy.strip() for allergy in str(x).split(',')]
        )
        
        # One-hot encode SkinType
        skin_dummies = pd.get_dummies(users_processed['SkinType'], prefix='Skin')
        users_processed = pd.concat([users_processed, skin_dummies], axis=1)
        
        # Encode HormoneLevel
        hormone_map = {'Low': 0, 'Normal': 1, 'High': 2}
        users_processed['HormoneLevel_encoded'] = users_processed['HormoneLevel'].map(hormone_map)
        
        # Normalize numeric features
        scaler = StandardScaler()
        users_processed[['Age_norm', 'BMI_norm']] = scaler.fit_transform(
            users_processed[['Age', 'BMI']]
        )
        
        self.processed_users = users_processed
        return users_processed

    def preprocess_trends(self):
        """Preprocess trends data"""
        trends_processed = self.trends_df.copy()
        
        # ROBUSTNESS FIX: Fill missing categorical data
        trends_processed['Type'].fillna('General', inplace=True)

        # One-hot encode Trend Type
        type_dummies = pd.get_dummies(trends_processed['Type'], prefix='Type')
        trends_processed = pd.concat([trends_processed, type_dummies], axis=1)
        
        # Extract keywords/ingredients for potential analysis (if needed later)
        trends_processed['KeyIngredients'] = trends_processed['KeyIngredients'].astype(str).str.lower()
        
        self.processed_trends = trends_processed
        return trends_processed
    
    def create_combined_features(self):
        """Create combined features for ML model using vectorized cross-merge (PERFORMANCE FIX)"""
        if self.processed_users is None:
            self.preprocess_users()
        if self.processed_trends is None:
            self.preprocess_trends()
            
        # PERFORMANCE FIX: Use vectorized cross-merge instead of slow nested loops
        users_temp = self.processed_users.copy()
        trends_temp = self.processed_trends.copy()
        
        # Create a temporary key for cross merge
        users_temp['_key'] = 1
        trends_temp['_key'] = 1
        
        self.combined_features = pd.merge(
            users_temp, 
            trends_temp, 
            on='_key'
        ).drop('_key', axis=1)
        
        return self.combined_features
