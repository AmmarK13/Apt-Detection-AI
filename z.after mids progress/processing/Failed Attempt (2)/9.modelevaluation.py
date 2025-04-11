import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load model components
model_path = "/home/kay/Documents/Workspace-S25/SE/SeProject/APT DETECTION/z.after mids progress/ddos_model.pkl"
try:
    model_dict = joblib.load(model_path)
    model = model_dict['model']
    features = model_dict['feature_names']
    scaler = model_dict['scaler']
except Exception as e:
    print(f"❌ Model loading failed: {str(e)}")
    exit(1)

# Load and validate dataset
dataset_path = '/home/kay/Documents/Workspace-S25/SE/SeProject/APT DETECTION/z.after mids progress/dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv (5)'
try:
    df = pd.read_csv(dataset_path)
except FileNotFoundError:
    print(f"❌ Dataset not found at {dataset_path}")
    exit(1)

# Clean up column names by stripping any leading or trailing spaces
df.columns = df.columns.str.strip()

# Debugging: Check the columns of the dataset
print(f"Columns in dataset: {df.columns}")

# Validate features
missing_features = set(features) - set(df.columns)
if missing_features:
    print(f"❌ Missing required features: {missing_features}")
    exit(1)

# Ensure feature order matches training
try:
    X = df[features].copy()
except KeyError as e:
    print(f"❌ Feature mismatch: {str(e)}")
    exit(1)

# Handle missing values
if X.isnull().any().any():
    print("⚠️  NaN values detected. Applying median imputation...")
    X = X.fillna(X.median())

# Handle infinity values
if np.isinf(X.values).any():
    print("⚠️  Infinity values detected. Replacing with max finite values...")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X = X.fillna(X.max())

# Validate labels
if 'Label' not in df.columns:
    print("❌ Missing 'Label' column in dataset")
    exit(1)

# Debugging: Check unique labels in the dataset
unique_labels = df['Label'].unique()
print(f"Unique labels in dataset: {unique_labels}")

# If only '0' labels are present (benign traffic), handle the case
if len(np.unique(unique_labels)) == 1 and 0 in unique_labels:
    print("❌ Only Benign traffic detected in dataset. Unable to evaluate model.")
    exit(1)

# Update the label mapping to include 'DDoS' as an attack
label_mapping = {'BENIGN': 0, 'benign': 0, 'DDoS': 1, 'ddos': 1, 'ATTACK': 1, 'attack': 1, 'malicious': 1}

# Check if the labels in the dataset match the expected ones
y = df['Label'].map(label_mapping)

# Check if mapping was successful
if y.isnull().any():
    print(f"❌ Unrecognized labels found in the dataset. Unmapped values: {df['Label'][y.isnull()]}")
    exit(1)

# Verify class balance
print("\n=== Class Distribution ===")
print(y.value_counts())

if len(np.unique(y)) < 2:
    print("❌ Only one class present in labels")
    exit(1)

# Preprocess data
try:
    X_scaled = scaler.transform(X)
except ValueError as e:
    print(f"❌ Scaling failed: {str(e)}")
    exit(1)

# Predict
try:
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]
except Exception as e:
    print(f"❌ Prediction failed: {str(e)}")
    exit(1)

# Evaluation metrics
print("\n=== Evaluation Metrics ===")
print(classification_report(y, y_pred))
print(f"ROC AUC: {roc_auc_score(y, y_proba):.4f}")

# Confusion Matrix
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues',
            xticklabels=['Benign', 'DDoS Attack'],
            yticklabels=['Benign', 'DDoS Attack'])
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y, y_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y, y_proba):.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
