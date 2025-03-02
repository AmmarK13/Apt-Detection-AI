import pandas as pd
import numpy as np
from scipy.stats import zscore
import os

# === CONFIGURATION ===
file_path = r"C:\Users\user\.cache\kagglehub\datasets\ogguy11\apt-detection\versions\1\02-16-2018.csv"
final_path=r"D:\4th semester\SE\project\cleaneddata"  # Your dataset path
output_dir = os.path.dirname(final_path)  # Get the directory of the dataset file
threshold_range = np.arange(1.5, 3.6, 0.1)  # Testing thresholds from 1.5 to 3.5 (step 0.1)

# === STEP 1: LOAD DATA ===
print("🔄 Loading dataset...")
data = pd.read_csv(file_path, low_memory=False)

# 🔹 Convert columns to proper numeric format (fix dtype warnings)
for col in data.columns:
    try:
        data[col] = pd.to_numeric(data[col], errors='coerce')  # Convert numbers, set errors to NaN
    except:
        pass  # Ignore non-convertible columns like 'Label' and 'Timestamp'

# 🔹 Drop non-numeric columns (like Timestamp) for outlier detection
numeric_columns = data.select_dtypes(include=['number']).columns

# === STEP 2: OUTLIER DETECTION METHODS ===

# 🟢 IQR Method
def detect_outliers_iqr(df, columns):
    outlier_indices = set()
    
    for col in columns:
        Q1 = np.nanpercentile(df[col], 25)
        Q3 = np.nanpercentile(df[col], 75)
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
        z_scores = zscore(df[col], nan_policy='omit')  # Ignore NaN values in calculations
        outliers = np.where(abs(z_scores) > threshold)[0]
        outlier_indices.update(outliers)

    return list(outlier_indices)

# === STEP 3: FIND BEST THRESHOLD ===
best_threshold = None
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

    # Update best threshold if attack percentage improves and benign count remains reasonable
    if attack_percentage > best_attack_percentage and len(benign_outliers) < len(attack_outliers) * 0.10:  
        best_attack_percentage = attack_percentage
        best_threshold = threshold
        best_outlier_indices = all_outlier_indices

# === STEP 4: FINAL OUTLIER DETECTION USING BEST THRESHOLD ===
if best_threshold is not None:
    print(f"\n✅ Best Threshold Found: {best_threshold:.1f} (Attack Outlier Percentage: {best_attack_percentage:.2f}%)")
    outliers_df = data.loc[best_outlier_indices]

    # Separate attack and benign outliers
    attack_outliers = outliers_df[outliers_df["Label"] != "Benign"]
    benign_outliers = outliers_df[outliers_df["Label"] == "Benign"]

    # Remove benign outliers from the original dataset
    cleaned_data = data.drop(index=benign_outliers.index)

    # === STEP 5: SAVE RESULTS TO THE SAME DIRECTORY AS INPUT FILE ===
    cleaned_data.to_csv(os.path.join(output_dir, "02-16-2018_reducedoutliers.csv"), index=False)  # New dataset without benign outliers

    print("\n✅ Outliers analysis complete! Results saved in:")
    print(f"- {output_dir}\\outliers_detected.csv (All detected outliers)")
    print(f"- {output_dir}\\attack_outliers.csv (Only attack-related outliers)")
    print(f"- {output_dir}\\benign_outliers.csv (Benign outliers)")
    print(f"- {output_dir}\\cleaned_data.csv (Dataset with benign outliers removed)")
else:
    print("\n❌ No suitable threshold found. Try adjusting the range or checking data preprocessing.")
