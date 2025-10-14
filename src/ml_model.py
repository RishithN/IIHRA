import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore') # Suppress warnings during training

class MLModel:
    def __init__(self, rule_engine):
        self.rule_engine = rule_engine
        self.label_encoder = LabelEncoder()
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.risk_df_cleaned = None

    def _preprocess_features(self, X):
        """Preprocess categorical features for ML model"""
        # Encode categorical features
        X_processed = pd.get_dummies(X, columns=['SkinType', 'TrendType'])
        
        # Simple feature for allergy count (0 or 1 for presence)
        X_processed['HasAllergies'] = X_processed['Allergies'].apply(
            lambda x: 1 if str(x).lower() not in ('none', 'nan') else 0
        )
        X_processed.drop(columns=['Allergies'], inplace=True)
        return X_processed

    def prepare_training_data(self):
        """Prepare training data from rule-based results"""
        # Ensure rule engine is run to generate data
        risk_df = self.rule_engine.generate_risk_matrix()
        
        # Use rule-based results as training labels
        X = risk_df[['Age', 'BMI', 'SkinType', 'Allergies', 'TrendType']].copy()
        y = risk_df['RiskLevel']
        
        # Preprocess features
        X_processed = self._preprocess_features(X)
        
        # ROBUSTNESS FIX: Handle missing values and align labels
        # Drop rows with NaNs to ensure model training doesn't fail
        X_processed.dropna(inplace=True)
        y = y[X_processed.index] # Align y with cleaned X

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.risk_df_cleaned = risk_df.loc[X_processed.index]
        
        return X_processed, y_encoded, self.risk_df_cleaned

    def train_model(self, model_type='random_forest'):
        """Train the ML model"""
        X_processed, y_encoded, _ = self.prepare_training_data()
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_processed, y_encoded, test_size=0.2, random_state=42
        )
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError("Unsupported model type.")
            
        self.model.fit(self.X_train, self.y_train)
        
        # Evaluate
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred, output_dict=True, 
                                       target_names=self.label_encoder.classes_)
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        
        return accuracy, report, conf_matrix, self.model

    def predict_risk(self, user_features, trend_features):
        """Make a prediction (example placeholder for future use)"""
        if self.model is None:
            raise Exception("Model not trained yet.")
        # Logic to combine/preprocess user_features and trend_features needed here
        return self.model.predict(np.array([1, 0, 1])) # Placeholder logic
