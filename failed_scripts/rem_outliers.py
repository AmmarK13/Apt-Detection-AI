import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# === CONFIGURATION ===
file_path = r"C:\Users\user\.cache\kagglehub\datasets\ogguy11\apt-detection\versions\1\02-21-2018.csv"  # Replace with your dataset file path
cleaned_file_path = r"D:\4th semester\SE\project\cleaneddata\02-21-2018_reducedoutliers.csv"
  # Replace with desired output path
threshold = 1.9 # Z-score threshold for outliers

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
def detect_outliers_zscore(df, columns, threshold=3):
    outlier_indices = set()

    for col in columns:
        z_scores = zscore(df[col])
        outliers = np.where(abs(z_scores) > threshold)[0]
        outlier_indices.update(outliers)

    return list(outlier_indices)

# Detect outliers using both methods
iqr_outliers = detect_outliers_iqr(data, numeric_columns)
zscore_outliers = detect_outliers_zscore(data, numeric_columns, threshold)

# Combine unique outliers from both methods
all_outlier_indices = list(set(iqr_outliers + zscore_outliers))

print(f"🔍 Total outliers detected: {len(all_outlier_indices)}")

# Extract outlier rows
outliers_df = data.loc[all_outlier_indices]

# === STEP 3: CHECK IF OUTLIERS ARE ATTACKS ===
attack_outliers = outliers_df[outliers_df["Label"] != "Benign"]
benign_outliers = outliers_df[outliers_df["Label"] == "Benign"]

print(f"⚠️ Outliers labeled as attacks: {len(attack_outliers)}")
print(f"✅ Outliers labeled as benign: {len(benign_outliers)}")
print(f"📊 Percentage of outliers that are attacks: {len(attack_outliers) / len(outliers_df) * 100:.2f}%" if len(outliers_df) > 0 else "0%")

# === STEP 4: REMOVE BENIGN OUTLIERS AND CREATE CLEANED DATASET ===
data_cleaned = data.drop(benign_outliers.index)

# Save the cleaned dataset
data_cleaned.to_csv(cleaned_file_path, index=False)
print(f"✅ Cleaned dataset saved as: {cleaned_file_path}")



# === STEP 6: SAVE RESULTS TO FILES ===
outliers_df.to_csv("outliers_detected.csv", index=False)
attack_outliers.to_csv("attack_outliers.csv", index=False)

print("✅ Outliers analysis complete! Results saved as:")
print("- outliers_detected.csv (All detected outliers)")
print("- attack_outliers.csv (Only attack-related outliers)")
print(f"- Cleaned dataset: {cleaned_file_path} (No benign outliers)")
