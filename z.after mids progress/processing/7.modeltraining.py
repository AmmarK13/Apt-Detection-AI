import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, precision_recall_curve, 
    RocCurveDisplay, f1_score  # Added f1_score here
)
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
    
    return X, y, X.columns.tolist()  # Return feature names

# 2. Class-Balanced Data Splitting
def split_data(X, y):
    # Stratified split preserving attack/benign ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        stratify=y,
        random_state=42
    )
    
    # Scale features and preserve column names
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    
    return X_train, X_test, y_train, y_test, scaler

# 3. Model Training with Cross-Validation
def train_model(X_train, y_train):
    # Calculate class weight dynamically
    class_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    
    # Updated model parameters
    model = XGBClassifier(
        tree_method='hist',
        scale_pos_weight=class_ratio,
        eval_metric='logloss',
        enable_categorical=False  # Replaces use_label_encoder
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
def save_artifacts(model, scaler, params, feature_names, X_test, y_test):
    # Generate predictions for saving
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Save model with metadata
    joblib.dump({
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names
    }, MODEL_SAVE_PATH)
    
    # Save metadata
    metadata = {
        'training_date': datetime.now().isoformat(),
        'features': feature_names,
        'best_params': params,
        'metrics': {
            'roc_auc': roc_auc_score(y_test, y_proba),
            'f1_score': f1_score(y_test, y_pred)
        }
    }
    
    with open(f"{RESULTS_DIR}model_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

# Update main execution
if __name__ == "__main__":
    # Load data
    X, y, feature_names = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test, scaler = split_data(X, y)
    
    # Train model
    print("Starting model training...")
    best_model, best_params = train_model(X_train, y_train)
    print(f"Best parameters: {best_params}")
    
    # Evaluate
    evaluate_model(best_model, X_test, y_test)
    
    # Save artifacts with test data
    save_artifacts(best_model, scaler, best_params, feature_names, X_test, y_test)
    print(f"Model saved to {MODEL_SAVE_PATH}")