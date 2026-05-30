import os
import pickle
import pandas as pd
from data_loader import load_credit_data, get_train_test_split
from preprocessing import get_preprocessing_pipeline
from model import train_and_evaluate_models

def run_training_pipeline():
    """
    Main orchestration function to run the Credit Scoring model training pipeline:
    1. Loads application and historical credit records.
    2. Aligns and merges data, formulating target labels.
    3. Splits data into train and test sets using stratified partitioning.
    4. Retrieves scikit-learn preprocessing pipeline.
    5. Trains Logistic Regression, Decision Tree, and Random Forest.
    6. Compares metrics, selects the best model, and saves it.
    """
    print("=======================================================================")
    print("      LAUNCHING CREDIT SCORING MODEL MACHINE LEARNING PIPELINE        ")
    print("=======================================================================\n")
    
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Load data
    df_merged = load_credit_data(workspace_dir)
    
    # 2. Train/test split
    X_train, X_test, y_train, y_test = get_train_test_split(df_merged)
    
    # 3. Create preprocessing pipeline
    preprocessing_pipeline = get_preprocessing_pipeline()
    
    # 4. Train and evaluate classifiers
    df_metrics, best_pipeline, best_model_name = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, preprocessing_pipeline
    )
    
    # 5. Print a beautifully formatted summary table
    print("\n=======================================================")
    print("             MODEL PERFORMANCE COMPARISON              ")
    print("=======================================================")
    print(df_metrics.round(4).to_string())
    print("=======================================================\n")
    
    # 6. Serialize and save the best model pipeline
    model_save_path = os.path.join(workspace_dir, "credit_model.pkl")
    print(f"[INFO] Saving the best model ({best_model_name}) pipeline...")
    with open(model_save_path, 'wb') as f:
        pickle.dump(best_pipeline, f)
        
    print(f"[SUCCESS] Trained model serialized and saved to: {model_save_path}")
    print("\nYou can now run predictions using: python predict.py")

if __name__ == "__main__":
    run_training_pipeline()
