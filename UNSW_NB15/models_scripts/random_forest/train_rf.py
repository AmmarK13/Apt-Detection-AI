import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load training and validation data
X_train = pd.read_csv('UNSW_NB15/prepared_data/X_train.csv')
y_train = pd.read_csv('UNSW_NB15/prepared_data/y_train.csv').values.ravel()
X_val = pd.read_csv('UNSW_NB15/prepared_data/X_val.csv')
y_val = pd.read_csv('UNSW_NB15/prepared_data/y_val.csv').values.ravel()
print("data loaded")

# Initialize and train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Validate the model
y_pred = rf_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f'Validation Accuracy: {accuracy:.4f}')

# Save the trained model
joblib.dump(rf_model, 'UNSW_NB15/models/model_rf.pkl')
