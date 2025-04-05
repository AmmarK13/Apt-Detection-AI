import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder

def test_model():
    """
    Test the trained model on new transformed dataset
    """
    print("Loading transformed test data...")
    test_data_path = "D:\\University\\Software Engineering\\Project\\transformed\\transformed_data.csv"
    test_df = pd.read_csv(test_data_path)
    
    print("Loading trained model...")
    model_path = "d:\\University\\Software Engineering\\Project\\Apt-Detection-AI\\webAttacks\\results\\xgboost_model.pkl"
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Convert string columns to numeric
    print("Converting data types...")
    for column in test_df.columns:
        if test_df[column].dtype == 'object':
            if column == 'URL' or column == 'content':
                # Keep URL and content as they are
                continue
            elif column == 'classification':
                # Convert classification to int
                test_df[column] = pd.to_numeric(test_df[column], errors='coerce').fillna(0).astype(int)
            else:
                # Use LabelEncoder for other categorical columns
                le = LabelEncoder()
                test_df[column] = le.fit_transform(test_df[column].astype(str))
    
    # Prepare test data
    X_test = test_df.drop(['classification'], axis=1)  # Keep URL and content for now
    y_test = test_df['classification']
    
    # Get expected feature names from model
    expected_features = ['has_hex_encoding', 'content_length', 'url_length', 'content-type', 
                        'contains_sql_injection', 'URL', 'path_depth', 'param_count', 'has_base64', 
                        'contains_special_chars', 'query_length', 'content', 'Accept', 
                        'content_special_chars', 'contains_command_injection', 'content_numbers', 
                        'contains_xss', 'cookie']
    
    # Ensure all expected features exist
    for feature in expected_features:
        if feature not in X_test.columns:
            if feature in ['URL', 'content']:
                X_test[feature] = ''
            else:
                X_test[feature] = 0
            print(f"Added missing feature: {feature}")
    
    # Reorder columns to match expected order
    X_test = X_test[expected_features]
    
    # Ensure all features are numeric and handle inf/large values
    print("Cleaning data...")
    # Only convert numeric columns, keep URL and content as strings
    numeric_features = [col for col in X_test.columns if col not in ['URL', 'content']]
    for column in numeric_features:
        X_test[column] = pd.to_numeric(X_test[column], errors='coerce')
    
    # Handle large values with log transformation
    for column in numeric_features:
        values = X_test[column].values
        if np.isfinite(values).all() and values.size > 0:  # Check if values are finite and not empty
            if values.max() > 1e6 or values.min() < -1e6:
                if values.min() >= 0:
                    # Log transformation for positive values
                    X_test[column] = np.log1p(values)
                else:
                    # Min-max scaling for data with negative values
                    min_val = values.min()
                    max_val = values.max()
                    if min_val != max_val:
                        X_test[column] = (values - min_val) / (max_val - min_val)
    
    # Replace any remaining inf values
    X_test = X_test.replace([np.inf, -np.inf], np.finfo(np.float64).max / 2)
    
    # Fill any remaining NA/null values
    X_test = X_test.fillna(0)
    
    print("Feature ranges after normalization:")
    print(X_test[numeric_features].describe())
    
    # Create a copy without URL and content for DMatrix
    X_test_numeric = X_test.drop(['URL', 'content'], axis=1)
    
    # Convert to DMatrix format
    print("Converting to DMatrix format...")
    dtest = xgb.DMatrix(data=X_test_numeric, label=y_test)
    
    # Make predictions
    print("Making predictions...")
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Print results
    print("\nModel Performance on New Dataset:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Save results
    results_dir = "d:\\University\\Software Engineering\\Project\\Apt-Detection-AI\\webAttacks\\results\\new_test"
    os.makedirs(results_dir, exist_ok=True)
    
    # Add predictions to original dataframe
    test_df['predicted_class'] = y_pred
    test_df['attack_probability'] = y_pred_proba
    predictions_path = os.path.join(results_dir, "predictions.csv")
    test_df.to_csv(predictions_path, index=False)
    print(f"\nPredictions saved to: {predictions_path}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    plt.title('Confusion Matrix on New Dataset')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    cm_path = os.path.join(results_dir, "confusion_matrix_new.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix plot saved to: {cm_path}")

if __name__ == "__main__":
    test_model()