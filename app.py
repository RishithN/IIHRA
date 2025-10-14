import streamlit as st
import pandas as pd
import sys
import os

# Add src to path (assuming your files are in a 'src' directory)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from data_loader import DataLoader
from preprocessor import DataPreprocessor
from rule_engine import RuleEngine
from ml_model import MLModel
from visualizations import Visualizations

# CRITICAL PERFORMANCE FIX: Cache heavy functions (run once)
@st.cache_data(show_spinner="Loading data and running risk assessment...")
def load_and_process_data():
    """Loads data, runs Rule Engine, and trains ML model once."""
    data_loader = DataLoader()
    
    if not data_loader.load_data():
        return None, None, None, None
    
    # Initialize Rule Engine (NOTE: __init__ must be fixed in rule_engine.py)
    rule_engine = RuleEngine(
        data_loader.users_df,
        data_loader.trends_df,
        data_loader.ingredient_risk_df
    )
    
    # Run Rule Engine (The heaviest step)
    risk_df = rule_engine.generate_risk_matrix()
    
    # LOGIC FIX: Train ML Model here and cache the object
    ml_model = None
    if not risk_df.empty:
        try:
            ml_model = MLModel(rule_engine)
            # Train the model, which prepares data and stores the model internally
            accuracy, _, _, _ = ml_model.train_model(model_type='random_forest')
            print(f"✅ ML Model Trained and Cached! Accuracy: {accuracy:.3f}")
        except Exception as e:
            print(f"❌ Failed to train ML model during caching: {e}")

    return data_loader, rule_engine, ml_model, risk_df


class TrendWiseApp:
    def __init__(self):
        self.data_loader = None
        self.rule_engine = None
        self.ml_model = None
        self.visualizations = None
        self.risk_df = None

    def setup(self):
        st.set_page_config(
            page_title="TrendWise - Social Media Trend Safety",
            page_icon="🔍",
            layout="wide"
        )
        
        # PERFORMANCE FIX: Call the cached function
        self.data_loader, self.rule_engine, self.ml_model, self.risk_df = load_and_process_data()

        if self.data_loader is None or self.risk_df is None or self.risk_df.empty:
            st.error("Failed to load or process data. Check console for file path and constructor errors.")
            return False

        # Initialize Visualizations with the cached data
        self.visualizations = Visualizations(
            self.risk_df,
            self.data_loader.users_df,
            self.data_loader.trends_df
        )
        
        return True

    def render_sidebar(self):
        st.sidebar.title("🔍 TrendWise")
        st.sidebar.markdown(
            "First system to predict safety of trending social media health products using personal data."
        )

        user_ids = self.data_loader.users_df['UserID'].tolist()
        selected_user = st.sidebar.selectbox("👤 Select User", user_ids)

        trend_types = ['All'] + self.data_loader.trends_df['Type'].unique().tolist()
        selected_type = st.sidebar.selectbox("📊 Filter by Trend Type", trend_types)

        risk_levels = ['All', 'Safe', 'Medium Risk', 'High Risk']
        selected_risk = st.sidebar.selectbox("⚠ Filter by Risk Level", risk_levels)

        if self.ml_model and self.ml_model.model:
            st.sidebar.info(f"ML Model Trained. Accuracy: {self.ml_model.model.score(self.ml_model.X_test, self.ml_model.y_test):.3f}")

        return selected_user, selected_type, selected_risk

    def render_main_dashboard(self, selected_user, selected_type, selected_risk):
        st.title("🚀 TrendWise - Social Media Trend Safety Analyzer")

        col1, col2, col3, col4 = st.columns(4)

        total_assessments = len(self.risk_df)
        safe_count = len(self.risk_df[self.risk_df['RiskLevel'] == 'Safe'])
        medium_risk_count = len(self.risk_df[self.risk_df['RiskLevel'] == 'Medium Risk'])
        high_risk_count = len(self.risk_df[self.risk_df['RiskLevel'] == 'High Risk'])

        with col1:
            st.metric("Total Assessments", total_assessments)
        with col2:
            st.metric("Safe Trends", safe_count, delta_color="normal")
        with col3:
            st.metric("Medium Risk", medium_risk_count, delta_color="off")
        with col4:
            st.metric("High Risk", high_risk_count, delta_color="inverse")

        st.subheader("📊 Overall Risk Analysis")

        col_overall_1, col_overall_2 = st.columns([1, 1.5]) # Adjusted columns for better chart display

        with col_overall_1:
            risk_dist_chart = self.visualizations.create_risk_distribution_chart()
            st.plotly_chart(risk_dist_chart, use_container_width=True)

        with col_overall_2:
            risk_by_type_chart = self.visualizations.create_risk_by_trend_type()
            st.plotly_chart(risk_by_type_chart, use_container_width=True)
            
        st.divider()

        st.subheader(f"👤 User {selected_user} Analysis & Recommendations")
        
        col_user_1, col_user_2 = st.columns(2)
        with col_user_1:
            # Display user profile
            profile_chart = self.visualizations.create_user_profile_summary(selected_user)
            st.plotly_chart(profile_chart, use_container_width=True)

        with col_user_2:
            # Display Risk Heatmap
            st.markdown("##### Trend Risk Heatmap (All Users)")
            heatmap = self.visualizations.create_risk_heatmap()
            st.plotly_chart(heatmap, use_container_width=True)


        st.subheader("🎯 Personalized Recommendations")

        user_risk_df = self.risk_df[self.risk_df['UserID'] == selected_user]

        if selected_type != 'All':
            user_risk_df = user_risk_df[user_risk_df['TrendType'] == selected_type]

        if selected_risk != 'All':
            user_risk_df = user_risk_df[user_risk_df['RiskLevel'] == selected_risk]
            
        if user_risk_df.empty:
            st.warning("No trends matched your filters for this user.")
            return

        safe_tab, medium_tab, high_tab = st.tabs(["✅ Safe", "⚠ Medium Risk", "🚨 High Risk"])

        # Function to render trend details
        def render_trends(df, indicator, color_func):
            if df.empty:
                st.info(f"No {indicator} trends found.")
            else:
                for _, trend in df.sort_values(by='RiskScore', ascending=False).iterrows():
                    st.markdown(f"**{trend['TrendName']}** *({trend['TrendType']})*", unsafe_allow_html=True)
                    st.markdown(f"Ingredients: `{trend['KeyIngredients']}`")
                    st.markdown(f'<div style="color:{color_func}; font-weight:bold;">Risk Score: {trend["RiskScore"]} ({indicator})</div>', unsafe_allow_html=True)
                    st.divider()

        with safe_tab:
            safe_trends = user_risk_df[user_risk_df['RiskLevel'] == 'Safe']
            render_trends(safe_trends, "SAFE", 'green')

        with medium_tab:
            medium_trends = user_risk_df[user_risk_df['RiskLevel'] == 'Medium Risk']
            render_trends(medium_trends, "MEDIUM RISK", 'orange')

        with high_tab:
            high_trends = user_risk_df[user_risk_df['RiskLevel'] == 'High Risk']
            render_trends(high_trends, "HIGH RISK", 'red')
        
        st.subheader("🔬 Ingredient and Trend Deep Dive")
        
        col_deep_1, col_deep_2 = st.columns(2)
        
        with col_deep_1:
            st.markdown("##### Trends with Highest % of High-Risk Assessments")
            trend_chart, _ = self.visualizations.create_trend_analysis()
            st.plotly_chart(trend_chart, use_container_width=True)

        with col_deep_2:
            st.markdown("##### Top 15 Risky Ingredients (Avg Risk Score)")
            ingredient_chart, _ = self.visualizations.create_ingredient_risk_analysis()
            if ingredient_chart:
                st.plotly_chart(ingredient_chart, use_container_width=True)
            else:
                st.info("No ingredient data available for deep dive.")


    def run(self):
        if not self.setup():
            return

        selected_user, selected_type, selected_risk = self.render_sidebar()
        self.render_main_dashboard(selected_user, selected_type, selected_risk)


if __name__ == "__main__":
    # Ensure you are running this from your project root with:
    # streamlit run app.py
    app = TrendWiseApp()
    app.run()
