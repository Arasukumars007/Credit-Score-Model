import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.pipeline import Pipeline

def train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessing_pipeline):
    """
    Trains Logistic Regression, Decision Tree, and Random Forest models.
    Evaluates them using multiple classification metrics, plots curves,
    and returns a summary of the metrics and the best trained pipeline.
    """
    print("\n--- Training and Evaluating Classifiers ---")
    
    # Define candidate models (unweighted to yield calibrated probabilities)
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=42
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=10, min_samples_split=20, random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=150, max_depth=15, min_samples_split=10, random_state=42, n_jobs=-1
        )
    }
    
    metrics_summary = {}
    trained_pipelines = {}
    test_predictions = {}
    test_probabilities = {}
    
    # Train each model using the scikit-learn Pipeline
    for name, clf in models.items():
        print(f"Training {name}...")
        
        # Build complete pipeline (Preprocessing + Classifier)
        pipeline = Pipeline(steps=[
            ('preprocessing', preprocessing_pipeline),
            ('classifier', clf)
        ])
        
        # Fit model
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline
        
        # Predict on test set
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        # Apply standard bank risk threshold (tolerance: 15% probability of default)
        # If default probability >= 15%, applicant is rejected (1), else approved (0)
        y_pred = (y_prob >= 0.15).astype(int)
        
        test_predictions[name] = y_pred
        test_probabilities[name] = y_prob
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        metrics_summary[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': roc_auc
        }
        
        print(f"  {name} Trained. ROC-AUC: {roc_auc:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
        
    df_metrics = pd.DataFrame(metrics_summary).T
    
    # Determine the best model based on ROC-AUC (standard for credit scoring)
    best_model_name = df_metrics['ROC-AUC'].idxmax()
    best_pipeline = trained_pipelines[best_model_name]
    print(f"\n[BEST MODEL] Best Model Selected: {best_model_name} (ROC-AUC: {df_metrics.loc[best_model_name, 'ROC-AUC']:.4f})")
    
    # Generate visual evaluation plots
    plot_evaluation_curves(y_test, test_probabilities, test_predictions, best_model_name, df_metrics)
    
    return df_metrics, best_pipeline, best_model_name

def plot_evaluation_curves(y_test, test_probabilities, test_predictions, best_model_name, df_metrics):
    """
    Generates and saves professional evaluation plots:
    1. ROC Curves for all models
    2. Precision-Recall Curves for all models
    3. Confusion Matrix for the selected best model
    """
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # 1. ROC Curves
    for name, y_prob in test_probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = df_metrics.loc[name, 'ROC-AUC']
        axes[0].plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.3f})", lw=2)
    
    axes[0].plot([0, 1], [0, 1], 'k--', label="Random Guess", lw=1.5)
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].set_title('ROC Curves (Receiver Operating Characteristic)', fontsize=14, fontweight='bold')
    axes[0].legend(loc="lower right", frameon=True, fontsize=10)
    
    # 2. Precision-Recall Curves
    for name, y_prob in test_probabilities.items():
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        f1_score_val = df_metrics.loc[name, 'F1-Score']
        axes[1].plot(recall, precision, label=f"{name} (F1 = {f1_score_val:.3f})", lw=2)
        
    axes[1].set_xlabel('Recall (Sensitivity)', fontsize=12)
    axes[1].set_ylabel('Precision (Positive Predictive Value)', fontsize=12)
    axes[1].set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    axes[1].legend(loc="lower left", frameon=True, fontsize=10)
    
    # 3. Confusion Matrix for Best Model
    best_pred = test_predictions[best_model_name]
    cm = confusion_matrix(y_test, best_pred)
    
    # Standard format labels
    group_names = ['True Neg (Good)', 'False Pos', 'False Neg', 'True Pos (Bad)']
    group_counts = [f"{value:d}" for value in cm.flatten()]
    group_percentages = [f"{value:.2%}" for value in cm.flatten() / np.sum(cm)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    
    sns.heatmap(cm, annot=labels, fmt="", cmap='Blues', ax=axes[2], cbar=False, 
                annot_kws={"fontsize":11, "fontweight":"semibold"})
    axes[2].set_xlabel('Predicted Label', fontsize=12)
    axes[2].set_ylabel('Actual Label', fontsize=12)
    axes[2].set_xticklabels(['Good (0)', 'Bad (1)'], fontsize=10)
    axes[2].set_yticklabels(['Good (0)', 'Bad (1)'], fontsize=10)
    axes[2].set_title(f'Confusion Matrix: {best_model_name}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Ensure images directory exists in workspace
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(workspace_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    save_path = os.path.join(images_dir, "evaluation_plots.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[INFO] Saved visualization dashboard to: {save_path}")
