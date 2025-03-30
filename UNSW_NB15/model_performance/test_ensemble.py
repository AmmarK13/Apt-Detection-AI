import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Load trained models
rf_model = joblib.load("UNSW_NB15/models/model_rf.pkl")  
xgb_model = joblib.load("UNSW_NB15/models/model_xgb.pkl")  

# Load the cleaned testing dataset
test_data = pd.read_csv("UNSW_NB15/cleaned_data/testing_selected_features.csv") 

# Separate features and labels
X_test = test_data.drop(columns=["label", "attack_cat"])
y_test = test_data["label"]

# Get predictions from both models
rf_preds = rf_model.predict(X_test)
xgb_preds = xgb_model.predict(X_test)

# **Ensemble Method - Majority Voting**
ensemble_preds = np.round((rf_preds + xgb_preds) / 2)  # Majority Voting (0 or 1)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, ensemble_preds)
precision = precision_score(y_test, ensemble_preds)
recall = recall_score(y_test, ensemble_preds)
f1 = f1_score(y_test, ensemble_preds)
conf_matrix = confusion_matrix(y_test, ensemble_preds)

# Print results
print("Ensemble Model Testing - Random Forest + XGBoost")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}\n")
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_report(y_test, ensemble_preds))
