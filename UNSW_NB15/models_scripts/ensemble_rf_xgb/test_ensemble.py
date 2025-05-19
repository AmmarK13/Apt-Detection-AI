import os
import joblib
import pandas as pd
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Load the pre-split data
X_train = pd.read_csv("UNSW_NB15/prepared_data/X_train.csv")
y_train = pd.read_csv("UNSW_NB15/prepared_data/y_train.csv").squeeze()

X_val = pd.read_csv("UNSW_NB15/prepared_data/X_val.csv")
y_val = pd.read_csv("UNSW_NB15/prepared_data/y_val.csv").squeeze()

X_test = pd.read_csv("UNSW_NB15/prepared_data/X_test.csv")
y_test = pd.read_csv("UNSW_NB15/prepared_data/y_test.csv").squeeze()

print("✅ Data loaded successfully.")

# Model filename
model_filename =  "UNSW_NB15/models/model_ensemble.pkl"

# Check if saved model exists
if os.path.exists(model_filename):
    print("🔄 Loading saved ensemble model...")
    ensemble_model = joblib.load(model_filename)
    print("✅ Saved model loaded.")
else:
    # Define individual models
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)

    # Create an ensemble model using VotingClassifier
    ensemble_model = VotingClassifier(
        estimators=[('rf', rf_model), ('xgb', xgb_model)],
        voting='soft'  # Soft voting improves performance
    )

    # Train the ensemble model
    print("🚀 Training ensemble model...")
    ensemble_model.fit(X_train, y_train)
    print("✅ Ensemble training complete.")

    # Save the model
    joblib.dump(ensemble_model, model_filename)
    print(f"💾 Model saved as `{model_filename}`")

# **Validation Accuracy Check**
y_val_pred = ensemble_model.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)
precision = precision_score(y_val, y_val_pred)
recall = recall_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)
conf_matrix = confusion_matrix(y_val, y_val_pred)

print("\n✅ **Validation Performance** ✅")
print(f"Validation Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}\n")
print("Confusion Matrix (Validation Data):")
print(conf_matrix)
print("\nClassification Report (Validation Data):")
print(classification_report(y_val, y_val_pred))

# **Test Accuracy Check**
y_test_pred = ensemble_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_conf_matrix = confusion_matrix(y_test, y_test_pred)

print("\n🚀 **Test Performance** 🚀")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1 Score: {test_f1:.4f}\n")
print("Confusion Matrix (Test Data):")
print(test_conf_matrix)
print("\nClassification Report (Test Data):")
print(classification_report(y_test, y_test_pred))
