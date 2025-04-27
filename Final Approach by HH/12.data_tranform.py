import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import random
from sklearn.preprocessing import StandardScaler

# Load your trained model
model = joblib.load('trained_model.pkl')

# Features that the model expects
model_features = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std',
    'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
    'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
    'Min Packet Length', 'Max Packet Length', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
    'ACK Flag Count', 'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
    'Fwd Header Length.1', 'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes',
    'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd',
    'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
]

# Load the reference training data
reference_data = pd.read_csv('train_val.csv')

# Load new test data
test_data = pd.read_csv('test.csv')

# Handle labels first
if 'Label' in test_data.columns:
    # Drop rows where Label is NaN before processing features
    test_data = test_data.dropna(subset=['Label'])
    y_true = test_data['Label']
else:
    y_true = None

# Handle missing features by sampling from reference data
for col in model_features:
    if col not in test_data.columns:
        # Sample from reference data if column missing
        sampled_value = reference_data[col].dropna().sample(1).values[0]
        test_data[col] = sampled_value
    else:
        # Handle missing values in existing columns
        if test_data[col].isnull().sum() > 0:
            sampled_values = reference_data[col].dropna()
            if not sampled_values.empty:
                test_data[col] = test_data[col].apply(
                    lambda x: random.choice(sampled_values) if pd.isna(x) else x
                )
            else:
                test_data[col] = test_data[col].fillna(0)

# Keep only model features (ignore extra columns)
X_test = test_data[model_features].copy()

# Preprocessing pipeline
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.fillna(X_test.mean(), inplace=True)
X_test = X_test.clip(upper=1e10)
X_test = X_test.astype(np.float32)

# Scale features
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluation (only if we have labels)
if y_true is not None:
    # Ensure lengths match
    if len(y_true) == len(y_pred):
        acc = accuracy_score(y_true, y_pred)
        conf = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred)

        print(f"\n✅ Test Accuracy: {acc}\n")
        print("📉 Confusion Matrix:")
        print(conf)
        print("\n🎯 Classification Report:")
        print(report)
    else:
        print("\n🔎 Unexpected mismatch after preprocessing")
else:
    print("\n🔎 No true labels found in testing data")
    # Save predictions
    output = test_data.copy()
    output['Predicted_Label'] = y_pred
    output.to_csv('predictions_with_labels.csv', index=False)
    print("📝 Predictions saved to 'predictions_with_labels.csv'")