import pandas as pd
import numpy as np

class RuleEngine:
    # CRITICAL FIX: Constructor name corrected from _init_ to __init__
    def __init__(self, users_df, trends_df, ingredient_risk_df):
        self.users_df = users_df
        self.trends_df = trends_df
        self.ingredient_risk_df = ingredient_risk_df
        self.risk_rules = self._define_risk_rules()

    def _define_risk_rules(self):
        """Define simple risk rules (unchanged)"""
        return {
            'Allergy': 3,
            'Skincare_Mismatch': 2,
            'Diet_Mismatch': 1,
            'Banned_Ingredient': 3
        }

    def check_allergy_risk(self, user_allergies, trend_ingredients):
        """Check for ingredients matching user allergies"""
        user_allergies = [a.strip().lower() for a in str(user_allergies).split(',') if str(a).strip()]
        trend_ingredients = [i.strip().lower() for i in str(trend_ingredients).split(',') if str(i).strip()]
        
        if any(ing in user_allergies for ing in trend_ingredients):
            return self.risk_rules['Allergy']
        return 0

    def check_skincare_risk(self, skin_type, trend_type, trend_ingredients):
        """Check skin type vs trend type/ingredients"""
        if trend_type == 'Skincare' and skin_type == 'Sensitive' and 'acid' in trend_ingredients.lower():
            return self.risk_rules['Skincare_Mismatch']
        return 0

    def check_diet_risk(self, bmi, trend_type, trend_ingredients):
        """Check high BMI vs weight-loss trends"""
        if trend_type == 'Dietary' and bmi > 30 and 'detox' in trend_ingredients.lower():
            return self.risk_rules['Diet_Mismatch']
        return 0

    def check_ingredient_risk_level(self, trend_ingredients):
        """Check ingredient against known high-risk list"""
        trend_ingredients = [i.strip().lower() for i in str(trend_ingredients).split(',') if str(i).strip()]
        high_risk_ingredients = self.ingredient_risk_df[self.ingredient_risk_df['RiskLevel'] == 'High']['Ingredient'].str.lower().tolist()
        
        if any(ing in high_risk_ingredients for ing in trend_ingredients):
            return self.risk_rules['Banned_Ingredient']
        return 0

    # NEW EFFICIENT METHOD: Takes row data directly for speed
    def _assess_risk_efficient(self, user, trend):
        """Assess overall risk for user-trend combination using row data."""
        risk_scores = []
        
        risk_scores.append(self.check_allergy_risk(user['Allergies'], trend['KeyIngredients']))
        risk_scores.append(self.check_skincare_risk(user['SkinType'], trend['Type'], trend['KeyIngredients']))
        risk_scores.append(self.check_diet_risk(user['BMI'], trend['Type'], trend['KeyIngredients']))
        risk_scores.append(self.check_ingredient_risk_level(trend['KeyIngredients']))
        
        max_risk = max(risk_scores)
        
        if max_risk >= 3:
            return "High Risk", max_risk
        elif max_risk >= 1:
            return "Medium Risk", max_risk
        else:
            return "Safe", max_risk

    def assess_risk(self, user_id, trend_name):
        """External API for single risk assessment (for debugging or single calls)"""
        user = self.users_df[self.users_df['UserID'] == user_id].iloc[0]
        trend = self.trends_df[self.trends_df['TrendName'] == trend_name].iloc[0]
        return self._assess_risk_efficient(user, trend)


    def generate_risk_matrix(self):
        """Generate risk assessment for all user-trend combinations efficiently."""
        risk_results = []
        
        # PERFORMANCE FIX: Iterate directly over rows (user_data/trend_data are Series)
        for _, user_data in self.users_df.iterrows():
            for _, trend_data in self.trends_df.iterrows():
                
                # Use the efficient assessment function
                risk_level, risk_score = self._assess_risk_efficient(user_data, trend_data)

                risk_results.append({
                    'UserID': user_data['UserID'],
                    'Age': user_data['Age'],
                    'BMI': user_data['BMI'],
                    'SkinType': user_data['SkinType'],
                    'Allergies': user_data['Allergies'],
                    'HormoneLevel': user_data['HormoneLevel'], # Include HormoneLevel
                    'TrendName': trend_data['TrendName'],
                    'TrendType': trend_data['Type'],
                    'KeyIngredients': trend_data['KeyIngredients'],
                    'RiskLevel': risk_level,
                    'RiskScore': risk_score
                })
        
        return pd.DataFrame(risk_results)
