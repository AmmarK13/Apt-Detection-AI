import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load the training+validation data
df = pd.read_csv('train_val.csv')

# Split features and labels
X = df.drop('Label', axis=1)
y = df['Label']

# Further split into train and validation (e.g., 80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate on validation set
y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f"✅ Validation Accuracy: {acc}\n")
print("📉 Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))
print("\n🎯 Classification Report:")
print(classification_report(y_val, y_pred))

# Save the model
joblib.dump(model, 'trained_model.pkl')
print("💾 Model saved as 'trained_model.pkl'")
