import xgboost as xgb
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load training and validation data
X_train = pd.read_csv('UNSW_NB15/prepared_data/X_train.csv')
Y_train = pd.read_csv('UNSW_NB15/prepared_data/y_train.csv').values.ravel()
X_val = pd.read_csv('UNSW_NB15/prepared_data/X_val.csv')
Y_val = pd.read_csv('UNSW_NB15/prepared_data/y_val.csv').values.ravel()
print("data loaded")


# Initialize XGBoost Classifier
xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",  
    eval_metric="logloss",  # Since it's classification
    use_label_encoder=False, 
    n_estimators=100,  # Number of trees (adjust as needed)
    learning_rate=0.1,  # Step size (adjust for tuning)
    max_depth=6,  # Depth of trees (affects complexity)
    random_state=42
)

# Train the model
print("training....")
xgb_model.fit(X_train, Y_train)
print("trained")

# Save the trained model
joblib.dump(xgb_model, "model_xgb.pkl")
print("Model saved as model_xgb.pkl")


# xx_val = X_val.drop(columns=["label", "attack_cat"])
# y_val = Y_val["label"]
# Predict on validation set
Y_val_pred = xgb_model.predict(X_val)



# Evaluate Performance
print("\nEvaluate Performance")
val_acc = accuracy_score(Y_val, Y_val_pred)
print(f"Validation Accuracy: {val_acc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(Y_val, Y_val_pred))

print("\nClassification Report:")
print(classification_report(Y_val, Y_val_pred))
