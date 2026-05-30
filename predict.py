import os
import pickle
import pandas as pd
import numpy as np

# Sample customer profiles for testing
SAMPLE_PROFILES = [
    {
        "Profile_Name": "Low Risk Customer (Highly Creditworthy)",
        "CODE_GENDER": "F",
        "FLAG_OWN_CAR": "Y",
        "FLAG_OWN_REALTY": "Y",
        "CNT_CHILDREN": 0,
        "AMT_INCOME_TOTAL": 350000.0,
        "NAME_INCOME_TYPE": "State servant",
        "NAME_EDUCATION_TYPE": "Higher education",
        "NAME_FAMILY_STATUS": "Married",
        "NAME_HOUSING_TYPE": "House / apartment",
        "DAYS_BIRTH": -18000,       # ~49 years old
        "DAYS_EMPLOYED": -4500,     # ~12 years employed
        "FLAG_WORK_PHONE": 0,
        "FLAG_PHONE": 1,
        "FLAG_EMAIL": 1,
        "OCCUPATION_TYPE": "Managers",
        "CNT_FAM_MEMBERS": 2.0
    },
    {
        "Profile_Name": "High Risk Customer (Likely Default)",
        "CODE_GENDER": "M",
        "FLAG_OWN_CAR": "N",
        "FLAG_OWN_REALTY": "N",
        "CNT_CHILDREN": 3,
        "AMT_INCOME_TOTAL": 45000.0,
        "NAME_INCOME_TYPE": "Working",
        "NAME_EDUCATION_TYPE": "Secondary / secondary special",
        "NAME_FAMILY_STATUS": "Single / not married",
        "NAME_HOUSING_TYPE": "Rented apartment",
        "DAYS_BIRTH": -8500,        # ~23 years old
        "DAYS_EMPLOYED": -180,      # ~6 months employed
        "FLAG_WORK_PHONE": 1,
        "FLAG_PHONE": 0,
        "FLAG_EMAIL": 0,
        "OCCUPATION_TYPE": "Laborers",
        "CNT_FAM_MEMBERS": 4.0
    },
    {
        "Profile_Name": "Retired Customer (Stable Income)",
        "CODE_GENDER": "F",
        "FLAG_OWN_CAR": "N",
        "FLAG_OWN_REALTY": "Y",
        "CNT_CHILDREN": 0,
        "AMT_INCOME_TOTAL": 120000.0,
        "NAME_INCOME_TYPE": "Pensioner",
        "NAME_EDUCATION_TYPE": "Secondary / secondary special",
        "NAME_FAMILY_STATUS": "Widow",
        "NAME_HOUSING_TYPE": "House / apartment",
        "DAYS_BIRTH": -23000,       # ~63 years old
        "DAYS_EMPLOYED": 365243,    # Unemployed / Retired flag
        "FLAG_WORK_PHONE": 0,
        "FLAG_PHONE": 1,
        "FLAG_EMAIL": 0,
        "OCCUPATION_TYPE": np.nan,  # Unknown / None
        "CNT_FAM_MEMBERS": 1.0
    }
]

def load_saved_model(model_path=None):
    """
    Loads the serialized best model pipeline from disk.
    """
    if model_path is None:
        workspace_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(workspace_dir, "credit_model.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"[ERROR] Saved model not found at {model_path}. "
            "Please run train.py first to train and serialize the model."
        )
        
    print(f"[INFO] Loading credit scoring model pipeline from: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_single_profile(model, profile_dict, risk_threshold=0.15):
    """
    Predicts the credit approval status and delinquency probability for a single profile dictionary.
    """
    # Create DataFrame from the single dictionary row
    df_single = pd.DataFrame([profile_dict])
    
    # Run prediction through the loaded pipeline
    prob_bad = model.predict_proba(df_single)[0, 1]
    prob_good = 1.0 - prob_bad
    
    # If default risk is >= risk_threshold, the applicant is rejected.
    pred_class_adjusted = 1 if prob_bad >= risk_threshold else 0
    
    # Calculate FICO-calibrated credit rating score based on risk (scale: 300 to 850)
    # 0% risk -> 850 score; 50%+ risk -> 300 score.
    credit_score = max(300, min(850, int(850 - (prob_bad * 1100))))
    
    decision = "APPROVED (Low Default Risk)" if pred_class_adjusted == 0 else "REJECTED (High Default Risk)"
    
    return {
        'Predicted_Class': int(pred_class_adjusted),
        'Probability_of_Default': float(prob_bad),
        'Probability_of_Approval': float(prob_good),
        'Credit_Score_Estimate': credit_score,
        'Decision': decision
    }

def run_sample_predictions():
    """
    Executes predictions on the embedded sample customer profiles.
    """
    try:
        model = load_saved_model()
    except FileNotFoundError as e:
        print(e)
        return
        
    print("\n=======================================================")
    print("      CREDIT WORTHINESS PREDICTIONS ON SAMPLE PROFILES  ")
    print("=======================================================\n")
    
    for i, profile in enumerate(SAMPLE_PROFILES, 1):
        name = profile['Profile_Name']
        # Remove label metadata for pure model input
        input_data = {k: v for k, v in profile.items() if k != "Profile_Name"}
        
        result = predict_single_profile(model, input_data)
        
        # Calculate human-readable age & employment
        age = int(-input_data['DAYS_BIRTH'] / 365.25)
        emp_days = input_data['DAYS_EMPLOYED']
        emp_status = "Retired/Unemployed" if emp_days > 0 else f"{int(-emp_days / 365.25)} years"
        
        print(f"Profile #{i}: {name}")
        print(f"   Details: Age: {age} | Job Type: {input_data['NAME_INCOME_TYPE']} ({input_data.get('OCCUPATION_TYPE', 'Unknown')})")
        print(f"            Employment: {emp_status} | Income: ${input_data['AMT_INCOME_TOTAL']:,.2f} | Education: {input_data['NAME_EDUCATION_TYPE']}")
        print(f"   -------------------------------------------------")
        print(f"   Credit Score Estimate : {result['Credit_Score_Estimate']} / 850")
        print(f"   Probability of Default: {result['Probability_of_Default']*100:.2f}%")
        print(f"   Probability of Approval: {result['Probability_of_Approval']*100:.2f}%")
        print(f"   Decision              : {result['Decision']}")
        print("=======================================================\n")

if __name__ == "__main__":
    run_sample_predictions()
