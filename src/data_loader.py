import pandas as pd
import numpy as np
import os
import sys

class DataLoader:
    def __init__(self):
        self.users_df = None
        self.trends_df = None
        self.ingredient_risk_df = None
        
        # Determine the directory where this script resides (the 'src' folder)
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

    def _get_data_path(self, filename):
        """Builds the path to access the 'data' folder which is one level up (../data)."""
        
        # FIX: We use os.pardir (which is '..') to move up one level from 'src'
        # to the project root, and then enter the 'data' folder.
        data_root = os.path.join(self.script_dir, os.pardir, 'data')
        
        final_path = os.path.join(data_root, filename)
        
        # Check if the file exists at the constructed path
        if os.path.exists(final_path):
            return final_path
        
        return None # File not found

    def load_data(self):
        """Load all datasets from CSV files using robust path resolution."""
        
        files_to_load = {
            'Users': 'Users.csv',
            'Trends': 'trends.csv',
            'Ingredient Risk': 'ingredient_risk.csv'
        }
        
        try:
            for name, filename in files_to_load.items():
                file_path = self._get_data_path(filename)
                
                if file_path is None:
                    # If any file is missing, raise a specific error with the filename
                    raise FileNotFoundError(f"Required file not found: {filename}")
                
                # Assign the loaded DataFrame to the correct attribute
                if name == 'Users':
                    self.users_df = pd.read_csv(file_path)
                elif name == 'Trends':
                    self.trends_df = pd.read_csv(file_path)
                elif name == 'Ingredient Risk':
                    self.ingredient_risk_df = pd.read_csv(file_path)
            
            print("✅ All data loaded successfully!")
            return True
            
        except FileNotFoundError as e:
            print(f"❌ DATA FILE ERROR: {e}")
            print("Please ensure the 'data' folder is directly outside the 'src' folder (e.g., IHHRA/data).")
            return False
        except Exception as e:
            print(f"❌ UNEXPECTED LOADING ERROR: {e}")
            return False
    
    def explore_data(self):
        """Basic data exploration summary (unchanged)"""
        if self.users_df is not None:
            print("Users Data Info:")
            self.users_df.info()
        if self.trends_df is not None:
            print("Trends Data Info:")
            self.trends_df.info()
        if self.ingredient_risk_df is not None:
            print("Ingredient Risk Data Info:")
            self.ingredient_risk_df.info()
    
    def get_data_summary(self):
        """Return summary statistics for numeric data (unchanged)"""
        summary = {}
        if self.users_df is not None:
            summary['users'] = self.users_df.describe().T
        if self.trends_df is not None:
            summary['trends'] = self.trends_df.describe().T
        return summary
