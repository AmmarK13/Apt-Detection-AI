import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# ========== INTEL OPTIMIZATION ==========
from sklearnex import patch_sklearn
patch_sklearn()  # Accelerates sklearn using Intel MKL

# ========== CONFIGURATION ==========
INPUT_FILE = r"Cleaned Dataset\02-14-2018.csv"  # 🟢 CHANGE INPUT PATH HERE
OUTPUT_DIR = r"Recleaned Dataset"               # 🟢 CHANGE OUTPUT DIRECTORY HERE
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n🔧 Configuration:")
print(f"Input: {INPUT_FILE}")
print(f"Output: {OUTPUT_DIR}\n")

# ========== STEP 1: DATA LOADING & PREPROCESSING ==========
print("🔄 [1/7] Loading and preprocessing data...")
try:
    data = pd.read_csv(INPUT_FILE, low_memory=False)
    print(f"✅ Loaded {len(data)} rows with {len(data.columns)} columns")
except Exception as e:
    print(f"❌ Failed to load data: {e}")
    exit()

# Preserve Label column separately
label_present = False
if 'Label' in data.columns:
    labels = data['Label'].copy()
    data = data.drop(columns=['Label'])
    label_present = True
    print("✅ 'Label' column preserved for later processing")
else:
    print("⚠️ No 'Label' column found - proceeding without class labels")

# Convert all columns to numeric
initial_cols = data.shape[1]
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data = data.dropna(axis=1, thresh=len(data)//2)
print(f"✅ Converted to numeric - Removed {initial_cols - data.shape[1]} non-numeric columns")

# ========== STEP 2: DATA CLEANING ==========
print("\n🔄 [2/7] Cleaning data...")
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.fillna(data.median(), inplace=True)
data = data.clip(lower=-1e9, upper=1e9)
print("✅ Removed Inf/NaN values and clipped extremes")

# Reattach labels after cleaning
if label_present:
    data['Label'] = labels

# ========== STEP 3: OUTLIER DETECTION ==========
print("\n🔄 [3/7] Detecting outliers...")

def detect_outliers(method, data):
    """Unified outlier detection function"""
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    if 'Label' in numeric_cols:
        numeric_cols.remove('Label')
    
    if method == 'iqr':
        outliers = set()
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers.update(data[(data[col] < (Q1 - 1.5*IQR)) | (data[col] > (Q3 + 1.5*IQR))].index)
        return list(outliers)
    
    elif method == 'zscore':
        outliers = set()
        for col in numeric_cols:
            z = np.abs(zscore(data[col]))
            outliers.update(np.where(z > 2.5)[0])
        return list(outliers)
    
    elif method == 'lof':
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
        return data[lof.fit_predict(data[numeric_cols]) == -1].index.tolist()
    
    elif method == 'isoforest':
        iso = IsolationForest(contamination=0.05, random_state=42)
        return data[iso.fit_predict(data[numeric_cols]) == -1].index.tolist()

# Run all detection methods
methods = {
    'IQR': detect_outliers('iqr', data),
    'Z-Score': detect_outliers('zscore', data),
    'LOF': detect_outliers('lof', data),
    'Isolation Forest': detect_outliers('isoforest', data)
}

# Combine results
all_outliers = list(set().union(*methods.values()))
outliers_df = data.loc[all_outliers]
print(f"✅ Total outliers detected: {len(all_outliers)} ({len(all_outliers)/len(data):.2%})")

# ========== STEP 4: OUTLIER PROCESSING ==========
print("\n🔄 [4/7] Processing outliers...")
if label_present:
    attack_outliers = outliers_df[outliers_df.Label != 'Benign']
    benign_outliers = outliers_df[outliers_df.Label == 'Benign']
    print(f"⚠️ Classification:\n- Attack: {len(attack_outliers)}\n- Benign: {len(benign_outliers)}")
else:
    attack_outliers = pd.DataFrame()
    benign_outliers = outliers_df
    print("⚠️ No labels - treating all outliers as benign")

# ========== STEP 5: DATA REFINEMENT ==========
print("\n🔄 [5/7] Refining dataset...")
if not benign_outliers.empty:
    keep = benign_outliers.sample(frac=0.05, random_state=42)
    cleaned_data = data.drop(benign_outliers.drop(keep.index).index)
else:
    cleaned_data = data.copy()

print(f"✅ Final dataset size: {len(cleaned_data)} rows")

# ========== STEP 6: SAVE RESULTS ==========
print("\n🔄 [6/7] Saving results...")
cleaned_data.to_csv(os.path.join(OUTPUT_DIR, 'cleaned_data.csv'), index=False)
outliers_df.to_csv(os.path.join(OUTPUT_DIR, 'all_outliers.csv'), index=False)

if label_present:
    attack_outliers.to_csv(os.path.join(OUTPUT_DIR, 'attack_outliers.csv'), index=False)
    benign_outliers.to_csv(os.path.join(OUTPUT_DIR, 'benign_outliers.csv'), index=False)

# ========== STEP 7: VISUALIZATION ==========
print("\n🔄 [7/7] Generating visualizations...")
plt.figure(figsize=(12, 6))
sns.boxplot(data=data.select_dtypes(include=np.number).iloc[:, :10])
plt.title('Feature Distribution Before Cleaning')
plt.xticks(rotation=45)
plt.savefig(os.path.join(OUTPUT_DIR, 'pre_clean.png'), bbox_inches='tight')

plt.figure(figsize=(12, 6))
sns.boxplot(data=cleaned_data.select_dtypes(include=np.number).iloc[:, :10])
plt.title('Feature Distribution After Cleaning')
plt.xticks(rotation=45)
plt.savefig(os.path.join(OUTPUT_DIR, 'post_clean.png'), bbox_inches='tight')

print("\n✅ Cleaning complete! Results saved to:", OUTPUT_DIR)
print("📊 Visualizations saved as 'pre_clean.png' and 'post_clean.png'")

