import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st
import numpy as np

class Visualizations:
    def __init__(self, risk_df, users_df, trends_df):
        self.risk_df = risk_df
        self.users_df = users_df
        self.trends_df = trends_df
    
    def create_risk_distribution_chart(self):
        """Create risk distribution pie chart (unchanged)"""
        if self.risk_df is None or self.risk_df.empty:
            return go.Figure().update_layout(title='No Risk Data Available')
            
        risk_counts = self.risk_df['RiskLevel'].value_counts()
        
        colors = {'Safe': 'green', 'Medium Risk': 'orange', 'High Risk': 'red'}
        
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title='Overall Risk Distribution',
            color=risk_counts.index,
            color_discrete_map=colors
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        return fig
    
    def create_risk_by_trend_type(self):
        """Create risk breakdown by trend type (unchanged)"""
        if self.risk_df is None or self.risk_df.empty:
            return go.Figure().update_layout(title='No Risk Data Available')

        risk_by_type = pd.crosstab(self.risk_df['TrendType'], self.risk_df['RiskLevel'])
        
        colors = {'Safe': 'green', 'Medium Risk': 'orange', 'High Risk': 'red'}
        
        fig = px.bar(
            risk_by_type,
            x=risk_by_type.index,
            y=risk_by_type.columns,
            title='Risk Distribution by Trend Type',
            barmode='stack',
            color_discrete_map=colors
        )
        fig.update_layout(xaxis_title='Trend Type', yaxis_title='Count')
        
        return fig
    
    def create_user_profile_summary(self, user_id):
        """Create user profile summary visualization (unchanged)"""
        user_data = self.users_df[self.users_df['UserID'] == user_id].iloc[0]
        
        fig = go.Figure()
        
        metrics_text = f"""
        <b>User Profile Summary</b><br>
        Age: {user_data['Age']}<br>
        BMI: {user_data['BMI']}<br>
        Skin Type: {user_data['SkinType']}<br>
        Allergies: {user_data['Allergies']}<br>
        Hormone Level: {user_data['HormoneLevel']}
        """
        
        fig.add_annotation(
            text=metrics_text,
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=14),
            align="left",
            bgcolor="lightblue"
        )
        
        fig.update_xaxes(showticklabels=False, range=[0, 1])
        fig.update_yaxes(showticklabels=False, range=[0, 1])
        fig.update_layout(
            title=f"User {user_id} Profile",
            showlegend=False,
            height=300
        )
        
        return fig
    
    def create_risk_heatmap(self):
        """Create heatmap of risk scores (LOGIC FIX)"""
        if self.risk_df is None or self.risk_df.empty:
            return go.Figure().update_layout(title='No Risk Data Available')

        # Pivot table using the NUMERIC 'RiskScore'
        heatmap_data_numeric = self.risk_df.pivot_table(
            index='UserID', 
            columns='TrendName', 
            values='RiskScore', 
            aggfunc='first'
        )
        
        # LOGIC FIX: The data is numeric (0, 1, 2). Use a custom color scale.
        fig = px.imshow(
            heatmap_data_numeric,
            title='Risk Heatmap: Users vs Trends',
            aspect='auto',
            # Custom scale defined for 0 (green), 1 (orange), 2 (red)
            color_continuous_scale=[[0, 'green'], [0.5, 'orange'], [1.0, 'red']], 
            zmin=0, # Set minimum value
            zmax=3 # Set maximum value (since max risk score is 3)
        )
        fig.update_layout(
            xaxis_title='Trends',
            yaxis_title='User ID'
        )
        
        return fig
    
    def create_trend_analysis(self):
        """Analyze which trends are most risky (unchanged)"""
        if self.risk_df is None or self.risk_df.empty:
            return go.Figure().update_layout(title='No Risk Data Available'), pd.DataFrame()

        # Calculate high risk percentage for each trend
        trend_risk = self.risk_df.groupby(['TrendName', 'TrendType']).agg({
            'RiskScore': ['mean', 'count'],
            'RiskLevel': lambda x: (x == 'High Risk').sum()
        }).round(3)
        
        trend_risk.columns = ['Avg_RiskScore', 'Total_Assessments', 'High_Risk_Count']
        trend_risk['High_Risk_Percentage'] = (trend_risk['High_Risk_Count'] / trend_risk['Total_Assessments'] * 100).round(1)
        
        fig = px.bar(
            trend_risk.reset_index(),
            x='TrendName',
            y='High_Risk_Percentage',
            color='TrendType',
            title='High Risk Percentage by Trend',
            labels={'High_Risk_Percentage': 'High Risk Cases (%)'}
        )
        fig.update_layout(xaxis_tickangle=-45)
        
        return fig, trend_risk.reset_index()
    
    def create_ingredient_risk_analysis(self):
        """Analyze risky ingredients across trends (PERFORMANCE FIX)"""
        if self.risk_df is None or self.risk_df.empty:
            return None, pd.DataFrame()
        
        # PERFORMANCE FIX: Use vectorized string splitting and explode instead of iterrows
        ingredient_df = self.risk_df[['RiskLevel', 'RiskScore', 'KeyIngredients']].copy()
        
        # Split ingredients string into a list and explode
        ingredient_df['Ingredient'] = ingredient_df['KeyIngredients'].astype(str).str.split(',')
        ingredient_df = ingredient_df.explode('Ingredient').copy()
        
        # Clean up ingredient names and filter out blanks
        ingredient_df['Ingredient'] = ingredient_df['Ingredient'].str.strip()
        ingredient_df = ingredient_df[ingredient_df['Ingredient'] != '']
        
        if ingredient_df.empty:
            return None, pd.DataFrame()
        
        ingredient_stats = ingredient_df.groupby('Ingredient').agg({
            'RiskScore': ['mean', 'count'],
            'RiskLevel': lambda x: (x == 'High Risk').sum()
        }).round(3)
        
        ingredient_stats.columns = ['Avg_RiskScore', 'Total_Occurrences', 'High_Risk_Count']
        ingredient_stats = ingredient_stats.sort_values('Avg_RiskScore', ascending=False)
        
        fig = px.bar(
            ingredient_stats.reset_index().head(15),
            x='Ingredient',
            y='Avg_RiskScore',
            color='Avg_RiskScore',
            title='Top 15 Risky Ingredients (Average Risk Score)',
            color_continuous_scale='reds'
        )
        fig.update_layout(xaxis_tickangle=-45)
        
        return fig, ingredient_stats.reset_index()
