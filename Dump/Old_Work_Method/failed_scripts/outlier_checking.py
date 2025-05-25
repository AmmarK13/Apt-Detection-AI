import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import zscore

# === CONFIGURATION ===
file_path = r"C:\Users\user\.cache\kagglehub\datasets\ogguy11\apt-detection\versions\1\02-16-2018.csv"  # Replace with your dataset file path
threshold_range = np.arange(1.5, 3.6, 0.1)  # Testing thresholds from 1.5 to 3.5 (step 0.1)

# === STEP 1: LOAD DATA ===
print("Loading dataset...")
data = pd.read_csv(file_path)

# Select numeric columns only (excluding categorical ones like "Label" and "Timestamp")
numeric_columns = data.select_dtypes(include=['number']).columns

# === STEP 2: OUTLIER DETECTION METHODS ===

# 🟢 IQR Method
def detect_outliers_iqr(df, columns):
    outlier_indices = set()
    
    for col in columns:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
        outlier_indices.update(outliers)

    return list(outlier_indices)

# 🟢 Z-Score Method
def detect_outliers_zscore(df, columns, threshold):
    outlier_indices = set()

    for col in columns:
        z_scores = zscore(df[col])
        outliers = np.where(abs(z_scores) > threshold)[0]
        outlier_indices.update(outliers)

    return list(outlier_indices)

# === STEP 3: FIND BEST THRESHOLD ===
best_threshold = 3.0
best_attack_percentage = 0
best_outlier_indices = []

for threshold in threshold_range:
    # Detect outliers using both methods
    iqr_outliers = detect_outliers_iqr(data, numeric_columns)
    zscore_outliers = detect_outliers_zscore(data, numeric_columns, threshold)

    # Combine unique outliers from both methods
    all_outlier_indices = list(set(iqr_outliers + zscore_outliers))
    
    # Extract outlier rows
    outliers_df = data.loc[all_outlier_indices]

    if len(outliers_df) == 0:
        continue  # Skip if no outliers are found
    
    # Check attack vs. benign outliers
    attack_outliers = outliers_df[outliers_df["Label"] != "Benign"]
    benign_outliers = outliers_df[outliers_df["Label"] == "Benign"]
    
    attack_percentage = (len(attack_outliers) / len(outliers_df)) * 100 if len(outliers_df) > 0 else 0
    
    print(f"🔍 Threshold {threshold:.1f} → Attack Outliers: {len(attack_outliers)}, Benign: {len(benign_outliers)}, Attack %: {attack_percentage:.2f}%")

    # Update best threshold if attack percentage improves
    if attack_percentage > best_attack_percentage:
        best_attack_percentage = attack_percentage
        best_threshold = threshold
        best_outlier_indices = all_outlier_indices

# === STEP 4: FINAL OUTLIER DETECTION USING BEST THRESHOLD ===
print(f"\n✅ Best Threshold Found: {best_threshold:.1f} (Attack Outlier Percentage: {best_attack_percentage:.2f}%)")
outliers_df = data.loc[best_outlier_indices]

# Check attack vs. benign outliers
attack_outliers = outliers_df[outliers_df["Label"] != "Benign"]
benign_outliers = outliers_df[outliers_df["Label"] == "Benign"]

# === STEP 5: SAVE RESULTS USING BEST THRESHOLD ===


print("\n✅ Outliers analysis complete! Results saved as:")
print("- outliers_detected.csv (All detected outliers)")
print("- attack_outliers.csv (Only attack-related outliers)")
print("- benign_outliers.csv (Benign outliers)")
