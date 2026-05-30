import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class CreditFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn transformer to perform feature engineering on credit records:
    - Calculates Age in Years from DAYS_BIRTH.
    - Handles negative/anomalous DAYS_EMPLOYED to create YEARS_EMPLOYED and IS_UNEMPLOYED flags.
    - Applies logarithmic scaling to AMT_INCOME_TOTAL.
    - Fills missing OCCUPATION_TYPE values with 'Unknown'.
    - Drops constant/uninformative columns (FLAG_MOBIL, ID) and raw source columns.
    """
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_out = X.copy()
        
        # 1. Transform Age (DAYS_BIRTH is negative)
        X_out['AGE_YEARS'] = -X_out['DAYS_BIRTH'] / 365.25
        
        # 2. Transform Employment (DAYS_EMPLOYED is negative for active, positive 365243 for unemployed)
        X_out['IS_UNEMPLOYED'] = (X_out['DAYS_EMPLOYED'] > 0).astype(int)
        X_out['YEARS_EMPLOYED'] = np.where(X_out['DAYS_EMPLOYED'] < 0, -X_out['DAYS_EMPLOYED'] / 365.25, 0.0)
        
        # 3. Log transformation of income to handle skewness
        X_out['LOG_INCOME'] = np.log1p(X_out['AMT_INCOME_TOTAL'])
        
        # 4. Handle missing values in Occupation
        if 'OCCUPATION_TYPE' in X_out.columns:
            X_out['OCCUPATION_TYPE'] = X_out['OCCUPATION_TYPE'].fillna('Unknown')
            
        # 5. Drop redundant and raw engineered columns
        cols_to_drop = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL', 'FLAG_MOBIL', 'ID']
        cols_to_drop = [col for col in cols_to_drop if col in X_out.columns]
        X_out = X_out.drop(columns=cols_to_drop)
        
        return X_out

def get_preprocessing_pipeline():
    """
    Constructs and returns the full scikit-learn preprocessing pipeline.
    This includes custom feature engineering, scaling, and categorical encoding.
    """
    # Columns expected after CreditFeatureTransformer runs
    num_cols = [
        'AGE_YEARS', 'YEARS_EMPLOYED', 'LOG_INCOME', 
        'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 
        'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL', 'IS_UNEMPLOYED'
    ]
    
    cat_cols = [
        'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 
        'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 
        'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE'
    ]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )
    
    full_pipeline = Pipeline(steps=[
        ('feature_engineering', CreditFeatureTransformer()),
        ('preprocessor', preprocessor)
    ])
    
    return full_pipeline

if __name__ == "__main__":
    # Test pipeline on a small mock dataset
    from data_loader import load_credit_data, get_train_test_split
    df = load_credit_data()
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    
    pipeline = get_preprocessing_pipeline()
    X_train_processed = pipeline.fit_transform(X_train)
    print("\nProcessed feature matrix shape:", X_train_processed.shape)
