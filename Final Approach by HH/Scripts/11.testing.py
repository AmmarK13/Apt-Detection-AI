import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the test data
test_df = pd.read_csv('test.csv')
X_test = test_df.drop('Label', axis=1)
y_test = test_df['Label']

# Load the trained model
model = joblib.load('trained_model.pkl')

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
acc = accuracy_score(y_test, y_pred)
print(f"✅ Test Accuracy: {acc}\n")
print("📉 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\n🎯 Classification Report:")
print(classification_report(y_test, y_pred))
