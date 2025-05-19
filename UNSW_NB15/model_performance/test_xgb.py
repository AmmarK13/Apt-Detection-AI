import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Load the trained XGBoost model
model = joblib.load("UNSW_NB15/models/model_xgb.pkl")  # Make sure this filename matches your saved XGBoost model

# Load the cleaned testing dataset
test_data = pd.read_csv("UNSW_NB15/cleaned_data/testing_selected_features.csv")  # Ensure correct path

# Separate features and labels
X_test = test_data.drop(columns=["label", "attack_cat"])
y_test = test_data["label"]

# Make predictions
y_pred = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print results
print("Testing data loaded.")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}\n")
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
