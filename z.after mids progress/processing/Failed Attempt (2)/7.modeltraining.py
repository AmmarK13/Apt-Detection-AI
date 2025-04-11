import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, precision_recall_curve, 
    RocCurveDisplay, f1_score, precision_score, recall_score
)
from xgboost import XGBClassifier
import joblib
import json
from datetime import datetime

# Configuration
DATA_PATH = "/home/kay/Documents/Workspace-S25/SE/SeProject/APT DETECTION/z.after mids progress/dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX_CLEANED_OPTIMAL_FEATURES.csv"
MODEL_SAVE_PATH = "/home/kay/Documents/Workspace-S25/SE/SeProject/APT DETECTION/z.after mids progress/ddos_model.pkl"
RESULTS_DIR = "/home/kay/Documents/Workspace-S25/SE/SeProject/APT DETECTION/z.after mids progress/"

def load_data():
    df = pd.read_csv(DATA_PATH)
    if df.isnull().any().any():
        raise ValueError("Dataset contains missing values")
    if 'Label' not in df.columns:
        raise KeyError("Label column missing from dataset")
    X = df.drop('Label', axis=1)
    y = df['Label']
    return X, y, X.columns.tolist()

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)
    return X_train, X_test, y_train, y_test, scaler

def train_model(X_train, y_train):
    class_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    base_model = XGBClassifier(
        tree_method='hist',
        eval_metric='logloss',
        enable_categorical=False,
        scale_pos_weight=class_ratio,
        use_label_encoder=False
    )
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='f1', n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    print("\n=== Key Metrics ===")
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")

    # Confusion Matrix
    plt.figure(figsize=(10, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
    plt.title('Confusion Matrix')
    plt.savefig(f"{RESULTS_DIR}confusion_matrix.png")
    plt.close()

    # ROC Curve
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title('ROC Curve')
    plt.savefig(f"{RESULTS_DIR}roc_curve.png")
    plt.close()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure()
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(f"{RESULTS_DIR}pr_curve.png")
    plt.close()

def save_artifacts(model, scaler, params, feature_names, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    feature_medians = pd.DataFrame(X_test, columns=feature_names).median().to_dict()
    joblib.dump({
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'feature_medians': feature_medians,
        'metadata': {
            'training_date': datetime.now().isoformat(),
            'best_params': params,
            'performance': {
                'roc_auc': roc_auc_score(y_test, y_proba),
                'f1_score': f1_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred)
            }
        }
    }, MODEL_SAVE_PATH)

    with open(f"{RESULTS_DIR}model_metadata.json", 'w') as f:
        json.dump({
            'training_date': datetime.now().isoformat(),
            'feature_count': len(feature_names),
            'best_parameters': params,
            'test_performance': {
                'roc_auc': float(roc_auc_score(y_test, y_proba)),
                'f1_score': float(f1_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred)),
                'recall': float(recall_score(y_test, y_pred))
            },
            'feature_list': feature_names
        }, f, indent=2)

if __name__ == "__main__":
    X, y, feature_names = load_data()
    X_train, X_test, y_train, y_test, scaler = split_data(X, y)
    print("🚀 Starting model training...")
    best_model, best_params = train_model(X_train, y_train)
    print(f"✅ Training complete! Best parameters: {best_params}")
    print("\n🔍 Model evaluation results:")
    evaluate_model(best_model, X_test, y_test)
    save_artifacts(best_model, scaler, best_params, feature_names, X_test, y_test)
    print(f"\n💾 Model saved to {MODEL_SAVE_PATH}")
    print(f"📊 Metadata saved to {RESULTS_DIR}model_metadata.json")
