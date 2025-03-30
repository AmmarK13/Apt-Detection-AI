import joblib
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Load training and validation data
X_train = pd.read_csv('UNSW_NB15/prepared_data/X_train.csv')
y_train = pd.read_csv('UNSW_NB15/prepared_data/y_train.csv').values.ravel()
X_val = pd.read_csv('UNSW_NB15/prepared_data/X_val.csv')
y_val = pd.read_csv('UNSW_NB15/prepared_data/y_val.csv').values.ravel()

# Drop attack_cat column if it exists
if "attack_cat" in X_train.columns:
    X_train = X_train.drop(columns=["attack_cat"])
if "attack_cat" in X_val.columns:
    X_val = X_val.drop(columns=["attack_cat"])

print("Data loaded and cleaned.")

# Define individual models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
xgb_model = XGBClassifier(eval_metric="logloss", random_state=42)

# Create an ensemble model using VotingClassifier
ensemble_model = VotingClassifier(
    estimators=[('rf', rf_model), ('xgb', xgb_model)],
    voting='soft'  # Use soft voting for better performance
)

# Train the ensemble model
print("Training ensemble model...")
ensemble_model.fit(X_train, y_train)
print("Ensemble training complete.")

# **Validation Accuracy Check**
y_val_pred = ensemble_model.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)
precision = precision_score(y_val, y_val_pred, average="macro")
recall = recall_score(y_val, y_val_pred, average="macro")
f1 = f1_score(y_val, y_val_pred, average="macro")
conf_matrix = confusion_matrix(y_val, y_val_pred)

# Print validation results
print("\n✅ **Validation Performance** ✅")
print(f"Validation Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}\n")

# Display Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=["No Attack", "Attack"], yticklabels=["No Attack", "Attack"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Save the model
joblib.dump(ensemble_model, "UNSW_NB15/models/model_ensemble.pkl")
print("Ensemble model saved as model_ensemble.pkl")