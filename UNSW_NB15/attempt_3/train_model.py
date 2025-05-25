import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv("UNSW_NB15/attempt_3/cleaned_attempt3/4_remove_features.csv")

# Separate features and label
X = df.drop(columns=["label"])
y = df["label"]

# Split into train and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on training data
y_train_pred = model.predict(X_train)

# Predict on test data
y_test_pred = model.predict(X_test)

# Evaluation
print("=== Training Set Results ===")
print("Accuracy:", accuracy_score(y_train, y_train_pred))
print("Classification Report:\n", classification_report(y_train, y_train_pred))
print("Confusion Matrix:\n", confusion_matrix(y_train, y_train_pred))

print("\n=== Test Set Results ===")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("Classification Report:\n", classification_report(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))


import joblib
import os

# Create output directory if it doesn't exist
output_dir = "UNSW_NB15/attempt_3/models/"
os.makedirs(output_dir, exist_ok=True)

# Save the trained model
model_path = os.path.join(output_dir, "random_forest_model.joblib")
joblib.dump(model, model_path)

print(f"Model saved to: {model_path}")
