from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pandas as pd
import joblib  # If you saved the model

# Load validation data
X_val = pd.read_csv('UNSW_NB15/prepared_data/X_val.csv')
y_val = pd.read_csv('UNSW_NB15/prepared_data/y_val.csv')
print("data loaded")


# Load trained model (if you saved it)
model = joblib.load("UNSW_NB15/models/model_rf.pkl")  # Update filename if needed
print("model loaded")


# Make predictions on validation set
y_pred = model.predict(X_val)

# Evaluation Metrics
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, average='weighted')
recall = recall_score(y_val, y_pred, average='weighted')
f1 = f1_score(y_val, y_pred, average='weighted')

# Confusion Matrix
conf_matrix = confusion_matrix(y_val, y_pred)

# Print results
print(f"Validation Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)

# More detailed classification report
print("\nClassification Report:")
print(classification_report(y_val, y_pred))
