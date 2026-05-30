import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from predict import predict_single_profile

# Set webpage configuration
st.set_page_config(
    page_title="PrediCredit - AI Credit Scoring Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Visual Aesthetics CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0e1117 0%, #161a24 100%);
        color: #e0e6ed;
    }
    .stButton>button {
        background: linear-gradient(90deg, #1f77b4 0%, #00d2ff 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(31, 119, 180, 0.4);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 210, 255, 0.6);
        background: linear-gradient(90deg, #00d2ff 0%, #1f77b4 100%);
    }
    .card {
        background-color: rgba(30, 39, 51, 0.65);
        border-radius: 12px;
        padding: 1.8rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(8px);
        margin-bottom: 1.5rem;
    }
    .score-approved {
        background: radial-gradient(circle, #0f3d24 0%, #071f12 100%);
        border: 2px solid #28a745;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 0 20px rgba(40, 167, 69, 0.25);
    }
    .score-rejected {
        background: radial-gradient(circle, #3d1416 0%, #200a0b 100%);
        border: 2px solid #dc3545;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 0 20px rgba(220, 53, 69, 0.25);
    }
    h1, h2, h3 {
        font-family: 'Outfit', 'Inter', sans-serif;
        letter-spacing: -0.5px;
    }
    .highlight {
        color: #00d2ff;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Main Title & Subtitle
st.markdown("<h1 style='text-align: center; color: white;'>🛡️ PrediCredit</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #a0aec0;'>State-of-the-Art Artificial Intelligence Credit Scoring & Risk Assessment System</p>", unsafe_allow_html=True)
st.markdown("---")

workspace_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(workspace_dir, "credit_model.pkl")
plots_path = os.path.join(workspace_dir, "images", "evaluation_plots.png")

# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/nolan/128/security-shield.png", width=100)
st.sidebar.markdown("## Navigation")
app_mode = st.sidebar.radio("Go to:", ["1. Credit Score Evaluator", "2. Model Architecture & Metrics"])

st.sidebar.markdown("---")
st.sidebar.markdown("## ⚙️ Underwriting Policy")
st.sidebar.write("Set the maximum tolerable default probability. Candidates exceeding this risk boundary are automatically rejected.")

risk_threshold_pct = st.sidebar.slider(
    "Max Default Risk Allowed (%)", 
    min_value=5.0, 
    max_value=50.0, 
    value=15.0, 
    step=1.0,
    help="Applicants with predicted probability of default above this value are rejected. Standard bank policies set this between 12% and 18%."
)
risk_threshold = risk_threshold_pct / 100.0

@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

model = load_model(model_path)

if app_mode == "1. Credit Score Evaluator":
    st.markdown("### 📋 Client Application & Risk Evaluator")
    st.write("Fill out the applicant's financial and demographic profile below to calculate their creditworthiness index and approval probability.")
    
    if model is None:
        st.warning("⚠️ Model file (`credit_model.pkl`) not found! Please run the training pipeline first using `python train.py` in your terminal to train and serialize the model.")
        
        # Fallback explanation or demo mode
        st.info("💡 Once trained, this form will connect directly to your Random Forest model pipeline.")
    
    # Input Form using layout columns for professional compact design
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("#### 👤 Applicant Demographics & Housing")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"], index=0)
        gender_code = "F" if gender == "Female" else "M"
        
        age_years = st.slider("Age (Years)", 18, 80, 35)
        days_birth = -int(age_years * 365.25)
        
    with col2:
        family_status = st.selectbox("Marital Status", [
            "Married", "Single / not married", "Civil marriage", "Separated", "Widow"
        ])
        
        housing_type = st.selectbox("Housing Type", [
            "House / apartment", "With parents", "Municipal apartment", 
            "Rented apartment", "Office apartment", "Co-op apartment"
        ])
        
    with col3:
        children = st.number_input("Number of Children", min_value=0, max_value=15, value=0)
        fam_members = st.number_input("Total Family Size", min_value=1, max_value=20, value=children + 2 if family_status == "Married" else children + 1)

    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("#### 💼 Employment & Wealth Parameters")
    col4, col5, col6 = st.columns(3)
    
    with col4:
        income_type = st.selectbox("Income Category", [
            "Working", "Commercial associate", "Pensioner", "State servant", "Student"
        ])
        
        annual_income = st.number_input("Annual Income ($)", min_value=1000.0, max_value=5000000.0, value=150000.0, step=5000.0)
        
    with col5:
        employment_status = st.selectbox("Employment Status", ["Employed", "Retired / Unemployed"])
        
        if employment_status == "Employed":
            years_employed = st.slider("Years Employed", 0.0, 50.0, 5.0, step=0.5)
            days_employed = -int(years_employed * 365.25)
        else:
            days_employed = 365243  # Unemployed/Retired flag used in dataset
            
    with col6:
        # Match values to occupation categories in dataset
        occupation = st.selectbox("Occupation Sub-type", [
            "Unknown", "Laborers", "Core staff", "Sales staff", "Managers", "Drivers",
            "High skill tech staff", "Accountants", "Medicine staff", "Security staff",
            "Cooking staff", "Cleaning staff", "Private service staff", "Low-skill Laborers",
            "Waiters/barmen staff", "Secretaries", "HR staff", "IT staff", "Realty agents"
        ])
        occupation_code = np.nan if occupation == "Unknown" else occupation
        
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("#### 🚗 Asset Ownership & Communication Flags")
    col7, col8, col9 = st.columns(3)
    
    with col7:
        own_car = st.selectbox("Owns Automobile?", ["No", "Yes"])
        own_car_code = "Y" if own_car == "Yes" else "N"
        
    with col8:
        own_realty = st.selectbox("Owns Real Estate/Property?", ["Yes", "No"])
        own_realty_code = "Y" if own_realty == "Yes" else "N"
        
    with col9:
        work_phone = st.checkbox("Work Phone Available", value=False)
        phone = st.checkbox("Home/Mobile Phone Available", value=True)
        email = st.checkbox("Email Address Provided", value=False)
        
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Assess button
    st.markdown("<div style='text-align: center; margin-top: 1.5rem;'>", unsafe_allow_html=True)
    assess_btn = st.button("📊 Assess Creditworthiness Index")
    st.markdown("</div>", unsafe_allow_html=True)
    
    if assess_btn:
        if model is None:
            st.error("❌ Cannot assess. Model pipeline is not trained yet. Please run `python train.py` in your terminal and refresh.")
        else:
            # Build input dictionary conforming to training columns
            applicant_profile = {
                "CODE_GENDER": gender_code,
                "FLAG_OWN_CAR": own_car_code,
                "FLAG_OWN_REALTY": own_realty_code,
                "CNT_CHILDREN": int(children),
                "AMT_INCOME_TOTAL": float(annual_income),
                "NAME_INCOME_TYPE": income_type,
                "NAME_EDUCATION_TYPE": "Higher education" if occupation in ["Managers", "Accountants", "IT staff", "HR staff"] else "Secondary / secondary special", # Simulated education mapping based on occupation
                "NAME_FAMILY_STATUS": family_status,
                "NAME_HOUSING_TYPE": housing_type,
                "DAYS_BIRTH": days_birth,
                "DAYS_EMPLOYED": days_employed,
                "FLAG_WORK_PHONE": 1 if work_phone else 0,
                "FLAG_PHONE": 1 if phone else 0,
                "FLAG_EMAIL": 1 if email else 0,
                "OCCUPATION_TYPE": occupation_code,
                "CNT_FAM_MEMBERS": float(fam_members)
            }
            
            with st.spinner("Processing profile through AI pipelines..."):
                results = predict_single_profile(model, applicant_profile, risk_threshold)
                
            st.markdown("---")
            st.markdown("### 🏁 Risk Assessment Results")
            
            card_class = "score-approved" if results['Predicted_Class'] == 0 else "score-rejected"
            
            col_res1, col_res2 = st.columns([1, 1])
            
            with col_res1:
                st.markdown(f"""
                    <div class='{card_class}'>
                        <h2 style='color: white; margin-top: 0;'>{results['Decision']}</h2>
                        <hr style='border-color: rgba(255,255,255,0.1);'>
                        <p style='font-size: 1rem; margin-bottom: 0.5rem;'>ESTIMATED CREDIT SCORE</p>
                        <h1 style='font-size: 4rem; margin-top: 0; color: white; font-weight: 800;'>{results['Credit_Score_Estimate']}</h1>
                        <p style='font-size: 0.9rem; color: #a0aec0;'>Scale: 300 (Highest Risk) to 850 (Highest Creditworthiness)</p>
                    </div>
                """, unsafe_allow_html=True)
                
            with col_res2:
                st.markdown("<div class='card' style='height: 100%;'>", unsafe_allow_html=True)
                st.write("#### 🛡️ AI Underwriting Probability Metrics")
                
                # Progress bars
                app_prob = results['Probability_of_Approval']
                def_prob = results['Probability_of_Default']
                
                st.write(f"**Probability of Good Credit Behavior:** {app_prob*100:.2f}%")
                st.progress(app_prob)
                
                st.write(f"**Probability of Delinquency (Default Risk):** {def_prob*100:.2f}%")
                st.progress(def_prob)
                
                st.write("#### 💡 Decision Risk Breakdown")
                if results['Predicted_Class'] == 0:
                    st.success("✅ **Approved**: The applicant's stable income, age, asset profile, or job description suggests low probability of late payments. Default risk is well within secure credit limits.")
                else:
                    st.error("❌ **Rejected**: The applicant exhibits delinquency patterns (potentially due to low income, short employment tenure, or high family size compared to income). The risk of a 30+ day overdue payment exceeds standard thresholds.")
                    
                st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("### 📊 Model Architecture & Performance Analytics")
    st.write("View the training metrics, comparative evaluation curves, and classification stats for the credit risk models.")
    
    if not os.path.exists(plots_path):
        st.warning("⚠️ Metrics dashboard plots not found! Please run the training orchestrator script (`python train.py`) first to generate performance curves.")
    else:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("#### 📈 Classifier Comparison Curves")
        st.write("These charts compare Logistic Regression, Decision Tree, and Random Forest on ROC space and Precision-Recall tradeoffs, alongside the Confusion Matrix of the chosen model.")
        
        # Display saved plot image
        st.image(plots_path, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("#### 💡 Understanding Model Metrics")
        st.markdown("""
        - **ROC-AUC (Area Under ROC Curve)**: Measures the model's ability to distinguish between Good and Bad credit risks across all thresholds. A score of **0.70+** is standard for demographic data, and higher indicates stronger predictive power.
        - **Precision-Recall Alignment**: Precision measures the ratio of actual defaults to predicted defaults, while Recall represents the percentage of actual default clients successfully detected.
        - **Calibrated Underwriting Thresholds**: Financial institutions rarely use a default machine learning threshold of 50% for credit approvals. Instead, they calibrate risk metrics based on economic constraints and set low risk tolerances (e.g. 12% to 18% max default probability). Any applicant exceeding this customized threshold is flagged for rejection or manual audit.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
