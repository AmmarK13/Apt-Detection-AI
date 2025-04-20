import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Path to your saved model
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attack_detection_rf_model.joblib")

def test_model(new_data_path, has_labels=True):
    """
    Test the trained model on new data and display performance metrics
    
    Args:
        new_data_path (str): Path to CSV file with new data
        has_labels (bool): Whether the data contains true labels ('classification' column)
    """
    # Load model and feature names
    model, feature_columns = joblib.load(MODEL_PATH)
    
    # Load new data
    new_data = pd.read_csv(new_data_path)
    
    # Verify required features exist
    missing_cols = [col for col in feature_columns if col not in new_data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Extract features
    X_new = new_data[feature_columns]
    
    # Make predictions
    predictions = model.predict(X_new)
    new_data['predicted_attack'] = predictions
    
    # Count attacks and normal requests
    attack_count = sum(predictions == 1)
    normal_count = sum(predictions == 0)
    total_count = len(predictions)
    
    # Calculate percentages
    attack_percentage = (attack_count / total_count) * 100
    normal_percentage = (normal_count / total_count) * 100
    
    print("\n=== Prediction Statistics ===")
    print(f"Total requests analyzed: {total_count}")
    print(f"Attacks detected: {attack_count} ({attack_percentage:.2f}%)")
    print(f"Normal requests: {normal_count} ({normal_percentage:.2f}%)")
    
    # If labels available, calculate metrics
    if has_labels and 'classification' in new_data.columns:
        y_true = new_data['classification']
        
        print("\n=== Model Performance ===")
        print(f"Accuracy: {accuracy_score(y_true, predictions):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, predictions))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, predictions))
    
    # Save predictions
    output_path = os.path.join(os.path.dirname(new_data_path), "original_dataset_predictions.csv")
    new_data.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")
    
    return new_data

# Example usage:
if __name__ == "__main__":
    # For data WITH labels:
    test_model("D:/University/Software Engineering/Project/data/csic_reduced_minmaxScaled.csv")
