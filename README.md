# TrendWise – Social Media Trend Safety Analyzer (IIHRA)

## Overview
TrendWise is an AI-powered and rule-based decision intelligence system that evaluates the safety of trending social media health, skincare, and dietary products using personalized user health data. The platform combines domain knowledge risk rules with machine learning prediction to generate accurate and explainable safety assessments for users. The system analyzes user attributes such as age, BMI, skin type, allergies, and hormone levels and compares them with trending product ingredients and categories to generate personalized safety recommendations. The goal of TrendWise is to reduce health risks caused by blindly following viral social media trends.

## Problem Statement
Social media frequently promotes skincare and dietary products without personalization. Many users follow these trends without understanding their compatibility with their body or health conditions. This can lead to allergic reactions, skin damage, hormonal imbalance, and other health risks due to unsafe or banned ingredients. TrendWise addresses this challenge by providing data-driven personalized safety analysis using artificial intelligence and rule-based medical logic.

## Key Features
TrendWise provides personalized risk assessment by evaluating each trend against each user's biological and health profile. It categorizes risk into Safe, Medium Risk, and High Risk levels. The system uses a hybrid intelligence model where a rule engine generates explainable baseline risk scores and a machine learning model learns hidden patterns from the generated dataset. The architecture is performance optimized using Streamlit caching, vectorized data preprocessing, and efficient risk evaluation loops. The dashboard provides interactive filtering, risk distribution visualization, heatmaps, ingredient-level analytics, and personalized recommendations.

## Technology Stack
The application is built using Streamlit and Plotly for frontend visualization and dashboard interaction. The backend uses Python with Pandas and NumPy for data processing. Machine learning is implemented using Scikit-Learn with a Random Forest Classifier. Feature engineering techniques such as One-Hot Encoding and Label Encoding are used to prepare data for model training.

## System Working
The user interacts with the Streamlit dashboard interface. The Data Loader module loads user data, trend data, and ingredient risk datasets. The Rule Engine evaluates risk for every user-trend combination using allergy checks, skincare compatibility checks, diet compatibility checks, and banned ingredient detection. The generated risk matrix is used as training data for the machine learning model. The trained model predicts risk levels and displays accuracy. The Visualization Engine generates interactive charts and analytics dashboards. The system then provides personalized safety recommendations for each user.

## Project Structure
IIHRA/
├── app.py
├── src/
│ ├── data_loader.py
│ ├── preprocessor.py
│ ├── rule_engine.py
│ ├── ml_model.py
│ └── visualizations.py
├── data/
│ ├── Users.csv
│ ├── trends.csv
│ └── ingredient_risk.csv
├── requirements.txt
└── README.md


## Installation and Setup
Clone the repository using Git:
git clone https://github.com/RishithN/IIHRA.git
cd IIHRA


Create a virtual environment:
python -m venv venv


Activate virtual environment:

Windows:
venv\Scripts\activate


Linux or Mac:
source venv/bin/activate


Install dependencies:
pip install -r requirements.txt


## Running the Application
Run the application from the project root folder:
streamlit run app.py


Then open the browser and navigate to:
http://localhost:8501


## Data Requirements
All datasets must be placed inside the data folder located in the project root directory. The required datasets include Users.csv, trends.csv, and ingredient_risk.csv. These datasets are used to generate the risk matrix and train the machine learning model.

## Machine Learning Workflow
The rule engine first generates risk labels based on domain knowledge rules. These labels are used as training targets for the machine learning model. Features such as age, BMI, skin type, allergies, and trend type are encoded and cleaned. The Random Forest model is trained using an 80-20 train test split. Model accuracy and performance metrics are displayed in the dashboard.

## Performance Optimization
The system uses Streamlit data caching to prevent repeated heavy computations. Vectorized Pandas operations improve preprocessing speed. Efficient nested risk evaluation is used instead of slow row-by-row computations. The machine learning model is trained once and reused for predictions.

## Future Scope
Future enhancements include deep learning based risk prediction, real-time social media trend scraping, NLP-based ingredient extraction, mobile app integration, and explainable AI risk reasoning modules.

## Use Cases
TrendWise can be used in health awareness platforms, dermatology recommendation systems, fitness and nutrition apps, preventive healthcare analytics platforms, and consumer safety monitoring systems.

## Author
Rishith N  
B.Tech Computer Science and Engineering (Data Analytics Specialization)  
Aspiring Data Analyst and Future Data Scientist

## License
This project is intended for academic and research purposes.
