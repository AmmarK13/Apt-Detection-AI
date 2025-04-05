import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, precision_recall_curve, RocCurveDisplay)
from xgboost import XGBClassifier
import joblib
import json
from datetime import datetime

# Configuration
DATA_PATH = "/home/kay/Documents/Workspace-S25/SE/SeProject/APT DETECTION/z.after mids progress/dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX_CLEANED_OPTIMAL_FEATURES.csv"
MODEL_SAVE_PATH = "/home/kay/Documents/Workspace-S25/SE/SeProject/APT DETECTION/z.after mids progress/ddos_model.pkl"
RESULTS_DIR = "/home/kay/Documents/Workspace-S25/SE/SeProject/APT DETECTION/z.after mids progress/"

# 1. Data Loading & Preparation
def load_data():
    df = pd.read_csv(DATA_PATH)
    
    # Final sanity checks
    assert not df.isnull().any().any(), "NaN values present in dataset"
    assert 'Label' in df.columns, "Label column missing"
    
    X = df.drop('Label', axis=1)
    y = df['Label']
    
    return X, y

# 2. Class-Balanced Data Splitting
def split_data(X, y):
    # Stratified split preserving attack/benign ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        stratify=y,
        random_state=42
    )
    
    # Optional: Scale features (XGBoost doesn't require, but useful for comparison)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler

# 3. Model Training with Cross-Validation
def train_model(X_train, y_train):
    # Base model with class weighting
    model = XGBClassifier(
        tree_method='hist',
        scale_pos_weight=1.75,  # Matches dataset imbalance
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    # Hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0]
    }
    
    # Stratified 5-fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Grid search
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    grid.fit(X_train, y_train)
    
    return grid.best_estimator_, grid.best_params_

# 4. Model Evaluation
def evaluate_model(model, X_test, y_test):
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
    
    # Confusion Matrix
    plt.figure(figsize=(8,6))
    sns.heatmap(confusion_matrix(y_test, y_pred), 
                annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Attack'],
                yticklabels=['Benign', 'Attack'])
    plt.title('Confusion Matrix')
    plt.savefig(f"{RESULTS_DIR}confusion_matrix.png")
    
    # ROC Curve
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title('ROC Curve')
    plt.savefig(f"{RESULTS_DIR}roc_curve.png")
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure()
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(f"{RESULTS_DIR}pr_curve.png")

# 5. Model Saving & Metadata
def save_artifacts(model, scaler, params):
    # Save model
    joblib.dump({'model': model, 'scaler': scaler}, MODEL_SAVE_PATH)
    
    # Save metadata
    metadata = {
        'training_date': datetime.now().isoformat(),
        'features': list(model.get_booster().feature_names),
        'best_params': params,
        'metrics': {
            'roc_auc': roc_auc_score(y_test, y_proba),
            'f1_score': f1_score(y_test, y_pred)
        }
    }
    
    with open(f"{RESULTS_DIR}model_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

# Main Execution
if __name__ == "__main__":
    # Load data
    X, y = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test, scaler = split_data(X, y)
    
    # Train model
    print("Starting model training...")
    best_model, best_params = train_model(X_train, y_train)
    print(f"Best parameters: {best_params}")
    
    # Evaluate
    evaluate_model(best_model, X_test, y_test)
    
    # Save artifacts
    save_artifacts(best_model, scaler, best_params)
    print(f"Model saved to {MODEL_SAVE_PATH}")