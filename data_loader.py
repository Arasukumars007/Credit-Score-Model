import pandas as pd
import os
from sklearn.model_selection import train_test_split

def load_credit_data(workspace_dir=None):
    """
    Loads raw application and credit record CSV files, matches client IDs,
    defines the target variable (30+ days delinquency as 'Bad' = 1, else 'Good' = 0),
    and returns a merged dataframe with both demographic features and the target.
    """
    if workspace_dir is None:
        workspace_dir = os.path.dirname(os.path.abspath(__file__))
    
    app_path = os.path.join(workspace_dir, "application_record.csv")
    credit_path = os.path.join(workspace_dir, "credit_record.csv")
    
    print("Loading datasets from:", workspace_dir)
    df_app = pd.read_csv(app_path)
    df_credit = pd.read_csv(credit_path)
    
    print(f"Loaded {df_app.shape[0]} application records and {df_credit.shape[0]} credit records.")
    
    # 1. Identify intersecting client IDs
    app_ids = set(df_app['ID'])
    credit_ids = set(df_credit['ID'])
    intersect_ids = app_ids.intersection(credit_ids)
    print(f"Number of intersecting clients: {len(intersect_ids)}")
    
    # Filter datasets to intersecting IDs
    df_app_filtered = df_app[df_app['ID'].isin(intersect_ids)].copy()
    df_credit_filtered = df_credit[df_credit['ID'].isin(intersect_ids)].copy()
    
    # Drop duplicates in application records (sometimes IDs can be duplicated in raw application files)
    df_app_filtered = df_app_filtered.drop_duplicates(subset=['ID'])
    print(f"Application records after dropping duplicates: {df_app_filtered.shape[0]}")
    
    # Re-align intersecting IDs after deduplication
    intersect_ids = intersect_ids.intersection(set(df_app_filtered['ID']))
    df_credit_filtered = df_credit_filtered[df_credit_filtered['ID'].isin(intersect_ids)]
    
    # 2. Define the Target Variable based on payment history
    # Delinquency numeric mapping:
    # 'C' (Paid off) and 'X' (No loan) -> -1 (No risk)
    # '0' (1-29 days past due) -> 0
    # '1' (30-59 days past due) -> 1
    # '2' (60-89 days past due) -> 2
    # '3' (90-119 days past due) -> 3
    # '4' (120-149 days past due) -> 4
    # '5' (150+ days overdue/write-off) -> 5
    def map_status(val):
        if val in ['C', 'X']:
            return -1
        return int(val)
    
    df_credit_filtered['STATUS_NUM'] = df_credit_filtered['STATUS'].apply(map_status)
    
    # A customer is "Bad" (1) if they ever had a payment overdue by 30+ days (STATUS_NUM >= 1)
    # Otherwise "Good" (0)
    client_max_status = df_credit_filtered.groupby('ID')['STATUS_NUM'].max()
    df_target = pd.DataFrame({
        'label': (client_max_status >= 1).astype(int)
    }).reset_index()
    
    # 3. Merge features and target
    df_merged = pd.merge(df_app_filtered, df_target, on='ID', how='inner')
    
    print("\n--- Target Variable Distribution ---")
    label_counts = df_merged['label'].value_counts()
    for label, count in label_counts.items():
        pct = (count / len(df_merged)) * 100
        status_name = "Bad (1)" if label == 1 else "Good (0)"
        print(f"  {status_name}: {count} ({pct:.2f}%)")
        
    return df_merged

def get_train_test_split(df, test_size=0.2, random_state=42):
    """
    Splits the merged dataset into stratified train and test sets.
    """
    X = df.drop(columns=['label'])
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nStratified Split Completed:")
    print(f"  Train set: {X_train.shape[0]} rows")
    print(f"  Test set:  {X_test.shape[0]} rows")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Test loading
    df = load_credit_data()
    X_train, X_test, y_train, y_test = get_train_test_split(df)
