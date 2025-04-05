import pandas as pd
import numpy as np
import xgboost as xgb
import os
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def train_model():
    """
    Train an XGBoost model for web attack detection using the prepared datasets.
    Evaluate model performance and save the trained model.
    """
    print("Starting model training process...")
    
    # Define paths
    data_dir = "d:\\University\\Software Engineering\\Project\\Apt-Detection-AI\\webAttacks\\data\\split"
    results_dir = "d:\\University\\Software Engineering\\Project\\Apt-Detection-AI\\webAttacks\\results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data - option 1: from CSV files
    print("Loading training and testing data...")
    train_df = pd.read_csv(os.path.join(data_dir, "train_data.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test_data.csv"))
    
    X_train = train_df.drop('classification', axis=1)
    y_train = train_df['classification']
    X_test = test_df.drop('classification', axis=1)
    y_test = test_df['classification']
    
    # Option 2: Load data from XGBoost binary format (faster)
    # Uncomment these lines to use the binary format instead
    # dtrain = xgb.DMatrix(os.path.join(data_dir, "dtrain.buffer"))
    # dtest = xgb.DMatrix(os.path.join(data_dir, "dtest.buffer"))
    
    # Convert to DMatrix format for XGBoost
    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    dtest = xgb.DMatrix(data=X_test, label=y_test)
    
    # Define model parameters
    params = {
        'objective': 'binary:logistic',  # Binary classification
        'eval_metric': 'logloss',        # Evaluation metric
        'eta': 0.1,                      # Learning rate
        'max_depth': 6,                  # Maximum tree depth
        'min_child_weight': 1,           # Minimum sum of instance weight needed in a child
        'subsample': 0.8,                # Subsample ratio of the training instances
        'colsample_bytree': 0.8,         # Subsample ratio of columns when constructing each tree
        'seed': 42                       # Random seed for reproducibility
    }
    
    # Train the model
    print("Training XGBoost model...")
    num_rounds = 100  # Number of boosting rounds
    
    # Train with early stopping
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_rounds,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=10,
        verbose_eval=10  # Print evaluation every 10 iterations
    )
    
    # Make predictions
    print("Evaluating model performance...")
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)  # Convert probabilities to binary predictions
    
    # Convert dtest labels to numpy array for evaluation
    y_test_np = dtest.get_label().astype(int)
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_test_np, y_pred)
    precision = precision_score(y_test_np, y_pred)
    recall = recall_score(y_test_np, y_pred)
    f1 = f1_score(y_test_np, y_pred)
    conf_matrix = confusion_matrix(y_test_np, y_pred)
    
    # Print results
    print("\nModel Performance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    print("\nClassification Report:")
    print(classification_report(y_test_np, y_pred))
    
    # Save the model
    model_path = os.path.join(results_dir, "xgboost_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to: {model_path}")
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    xgb.plot_importance(model, max_num_features=15, height=0.8)
    plt.title('Feature Importance (Weight)')
    plt.tight_layout()
    
    # Save the plot
    importance_path = os.path.join(results_dir, "feature_importance.png")
    plt.savefig(importance_path)
    print(f"Feature importance plot saved to: {importance_path}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    # Save the confusion matrix plot
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Confusion matrix plot saved to: {cm_path}")
    
    return model

if __name__ == "__main__":
    train_model()