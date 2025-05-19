import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# === CONFIGURATION ===
file_path = r"C:\Users\user\.cache\kagglehub\datasets\ogguy11\apt-detection\versions\1\02-22-2018.csv"
output_dir = r"D:\4th semester\SE\project\cleaneddata"
os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

# === STEP 1: LOAD DATA & CLEAN NON-NUMERIC COLUMNS ===
print("🔄 Loading dataset (This may take a while)...")
data = pd.read_csv(file_path, low_memory=False)

# 🔹 Convert columns to numeric (coerce errors to NaN)
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# 🔹 Drop non-numeric columns (like Timestamp)
data = data.drop(columns=['Timestamp'], errors='ignore')

# 🔹 Drop columns with >50% missing values
data = data.dropna(thresh=len(data) * 0.5, axis=1)

# 🔹 Fill remaining NaN values with column median
data.fillna(data.median(), inplace=True)

# 🔹 Define numeric columns
numeric_columns = data.select_dtypes(include=['number']).columns

# 🔹 Ensure no infinite values exist
data.replace([np.inf, -np.inf], np.nan, inplace=True)
if data[numeric_columns].isna().sum().sum() > 0:
    print("⚠️ Warning: NaN values detected! Filling them with median.")
    data.fillna(data.median(), inplace=True)

# === STEP 2: MULTI-METHOD OUTLIER DETECTION ===
print("🔍 Detecting outliers...")

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

# 🟢 Z-Score Method (Threshold Auto-Tuned)
def detect_outliers_zscore(df, columns, threshold=2.5):
    outlier_indices = set()
    for col in columns:
        z_scores = zscore(df[col], nan_policy='omit')
        outliers = np.where(abs(z_scores) > threshold)[0]
        outlier_indices.update(outliers)
    return list(outlier_indices)

# 🟢 Local Outlier Factor (LOF)
def detect_outliers_lof(df, columns, contamination=0.05):
    lof = LocalOutlierFactor(n_neighbors=50, contamination=contamination)
    outlier_pred = lof.fit_predict(df[columns])
    return list(df.index[outlier_pred == -1])  # Ensure proper indexing

# 🟢 Isolation Forest
def detect_outliers_isolation_forest(df, columns, contamination=0.05):
    iso_forest = IsolationForest(n_estimators=100, contamination=contamination, random_state=42, verbose=1)
    outlier_pred = iso_forest.fit_predict(df[columns])
    return list(df.index[outlier_pred == -1])

# === STEP 3: COMBINE OUTLIER DETECTION METHODS ===
iqr_outliers = detect_outliers_iqr(data, numeric_columns)
zscore_outliers = detect_outliers_zscore(data, numeric_columns)
lof_outliers = detect_outliers_lof(data, numeric_columns)
iso_forest_outliers = detect_outliers_isolation_forest(data, numeric_columns)

# Combine unique outliers from all methods
all_outlier_indices = list(set(iqr_outliers + zscore_outliers + lof_outliers + iso_forest_outliers))

print(f"✅ Total outliers detected: {len(all_outlier_indices)}")

# Extract outlier rows
outliers_df = data.loc[all_outlier_indices]

# === STEP 4: FILTER OUT ATTACK & BENIGN OUTLIERS ===
attack_outliers = outliers_df[outliers_df["Label"] != "Benign"]
benign_outliers = outliers_df[outliers_df["Label"] == "Benign"]

print(f"⚠️ Attack-related outliers: {len(attack_outliers)}")
print(f"✅ Benign outliers: {len(benign_outliers)}")
attack_percentage = (len(attack_outliers) / len(outliers_df) * 100) if len(outliers_df) > 0 else 0
print(f"📊 Attack Outlier Percentage: {attack_percentage:.2f}%")

# === STEP 5: REMOVE BENIGN OUTLIERS INTELLIGENTLY ===
# Keep 5% of benign outliers to avoid over-filtering
benign_to_keep = benign_outliers.sample(frac=0.05, random_state=42) if not benign_outliers.empty else pd.DataFrame()

# Remove benign outliers except the ones we keep
benign_to_remove = benign_outliers.drop(benign_to_keep.index)

# Final cleaned dataset
cleaned_data = data.drop(index=benign_to_remove.index)

# === STEP 6: SAVE RESULTS ===
cleaned_data.to_csv(os.path.join(output_dir, "02-22-2018_cleaned.csv"), index=False)
outliers_df.to_csv(os.path.join(output_dir, "outliers_detected.csv"), index=False)
attack_outliers.to_csv(os.path.join(output_dir, "attack_outliers.csv"), index=False)
benign_outliers.to_csv(os.path.join(output_dir, "benign_outliers.csv"), index=False)

print("\n✅ Outliers analysis complete! Results saved in:")
print(f"- {output_dir}/02-22-2018_cleaned.csv (Dataset with most benign outliers removed)")
print(f"- {output_dir}/outliers_detected.csv (All detected outliers)")
print(f"- {output_dir}/attack_outliers.csv (Only attack-related outliers)")
print(f"- {output_dir}/benign_outliers.csv (Benign outliers detected but some retained)")

# === STEP 7: PLOT OUTLIER DISTRIBUTIONS ===
plt.figure(figsize=(10, 5))
sns.boxplot(data=data[numeric_columns])
plt.xticks(rotation=90)
plt.title("Boxplot of Numeric Features (Before Outlier Removal)")
plt.savefig(os.path.join(output_dir, "before_outlier_removal.png"))
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(data=cleaned_data[numeric_columns])
plt.xticks(rotation=90)
plt.title("Boxplot of Numeric Features (After Outlier Removal)")
plt.savefig(os.path.join(output_dir, "after_outlier_removal.png"))
plt.show()
