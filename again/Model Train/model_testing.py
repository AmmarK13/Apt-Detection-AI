import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load new unbalanced dataset
file_path = r"C:\Users\user\.cache\kagglehub\datasets\ogguy11\apt-detection\versions\1\02-14-2018.csv"
df = pd.read_csv(file_path)

# Step 0: Convert 'Timestamp' column to Unix time (seconds since Unix epoch)
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', dayfirst=True)
df['Timestamp'] = (df['Timestamp'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

# Step 1: Handle missing values (drop rows with NaN values)
df.dropna(inplace=True)

# Step 2: Outlier Removal (Z-score method)
def filter_outliers_zscore(data, threshold):
    z_scores = np.abs(stats.zscore(data))
    outlier_mask = (z_scores > threshold).any(axis=1)
    return data[~outlier_mask]

threshold = 7
numerical_cols = [col for col in df.columns if col != 'Label']
df_filtered = filter_outliers_zscore(df[numerical_cols], threshold)

# Re-add the 'Label' column after filtering
df_filtered['Label'] = df.loc[df_filtered.index, 'Label']

# Step 3: Replace inf/-inf with NaN and drop those rows
df_filtered.replace([np.inf, -np.inf], np.nan, inplace=True)
df_filtered.dropna(inplace=True)

# Step 4: Drop the 16 features based on correlation analysis
columns_to_drop = [
    'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Min', 'Pkt Len Min', 'Pkt Len Max',
    'SYN Flag Cnt', 'ECE Flag Cnt', 'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg',
    'Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts',
    'Idle Max', 'Idle Min'
]
df_filtered.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# Step 5: Scale features
feature_cols = [col for col in df_filtered.columns if col != 'Label']
scaler = MinMaxScaler()
df_filtered[feature_cols] = scaler.fit_transform(df_filtered[feature_cols])

# Step 6: Split into X and y
X = df_filtered.drop('Label', axis=1)
y = df_filtered['Label'].astype(str)  # Ensure all labels are string type

# Step 7: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Load pre-trained models
rf_model = joblib.load(r"D:\4th semester\SE\project\Models\balanced_FTP-BruteForce_model.pkl")
svm_model = joblib.load(r"D:\4th semester\SE\project\Models\balanced_SSH-BruteForce_model.pkl")

# Step 9: Evaluation function
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test).astype(str)  # Convert predicted labels to strings
    y_test = y_test.astype(str)                 # Convert true labels to strings
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary', pos_label='Attack')
    recall = recall_score(y_test, y_pred, average='binary', pos_label='Attack')
    f1 = f1_score(y_test, y_pred, average='binary', pos_label='Attack')
    return accuracy, precision, recall, f1

# Step 10: Evaluate both models
print("\n==== Evaluating Random Forest Model ====")
rf_accuracy, rf_precision, rf_recall, rf_f1 = evaluate_model(rf_model, X_test, y_test)
print(f"Accuracy: {rf_accuracy:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall: {rf_recall:.4f}")
print(f"F1 Score: {rf_f1:.4f}")

print("\n==== Evaluating SVM Model ====")
svm_accuracy, svm_precision, svm_recall, svm_f1 = evaluate_model(svm_model, X_test, y_test)
print(f"Accuracy: {svm_accuracy:.4f}")
print(f"Precision: {svm_precision:.4f}")
print(f"Recall: {svm_recall:.4f}")
print(f"F1 Score: {svm_f1:.4f}")
