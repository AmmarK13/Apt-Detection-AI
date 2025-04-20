import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Define model path with absolute path
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attack_detection_rf_model.joblib")

def train_random_forest_model(input_csv_path):

    # Load the dataset
    data = pd.read_csv(input_csv_path)
    
    # Separate features and target
    X = data.drop('classification', axis=1) if 'classification' in data.columns else data.iloc[:, :-1]
    y = data['classification'] if 'classification' in data.columns else data.iloc[:, -1]
    
    # Get feature column names
    feature_columns = X.columns.tolist()
    
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    
    # Display metrics
    print("\nTraining Results:")
    print(f"- Training Accuracy: {train_accuracy:.4f}")
    print(f"- Test Accuracy:    {test_accuracy:.4f}")
    print(f"- Precision:        {precision:.4f}")
    print(f"- Recall:           {recall:.4f}")
    print(f"- F1 Score:         {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    
    # Save the model and feature columns
    joblib.dump((model, feature_columns), MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    
    return model, feature_columns

def predict_attack(input_csv_path, output_csv_path="D:/University/Software Engineering/Project/data/predictions.csv"):
    
    print(f"Running attack prediction on {input_csv_path}")
    
    # Check if model exists, train if it doesn't
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Training new model...")
        model, feature_columns = train_random_forest_model(input_csv_path)
    else:
        # Load the trained model and feature names
        print(f"Loading model from {MODEL_PATH}")
        model, feature_columns = joblib.load(MODEL_PATH)
    
    # Load new data
    new_data = pd.read_csv(input_csv_path)
    
    # Ensure the same columns as training data
    for col in feature_columns:
        if col not in new_data.columns:
            raise ValueError(f"Column {col} not found in input data")
    
    new_data_features = new_data[feature_columns]
    
    # Predict (0 = no attack, 1 = attack)
    predictions = model.predict(new_data_features)
    
    # Calculate accuracy if labels are available
    if 'classification' in new_data.columns:
        true_labels = new_data['classification']
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)
        
        print("\nPrediction Performance Metrics:")
        print(f"- Accuracy:  {accuracy:.4f}")
        print(f"- Precision: {precision:.4f}")
        print(f"- Recall:    {recall:.4f}")
        print(f"- F1 Score:  {f1:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(true_labels, predictions))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(true_labels, predictions))
    else:
        print("\nNo ground truth labels available. Only predictions generated.")
    
    # Add predictions to DataFrame
    new_data["predicted_attack"] = predictions
    
    # Save predictions
    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    new_data.to_csv(output_csv_path, index=False)
    print(f"\nPredictions saved to '{output_csv_path}'")
    
    return new_data

# Example: Predict on a new CSV file
if __name__ == "__main__":
    predict_attack("D:/University/Software Engineering/Project/data/minmaxScaled.csv")  # Replace with your input CSV path