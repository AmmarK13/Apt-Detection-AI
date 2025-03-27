# DataSets list:
- chethuhn/network-intrusion-dataset
- dhoogla/unswnb15
- spangler/csic-2010-web-application-attacks
- solarmainframe/ids-intrusion-csv
# Data Imported from Kaggle:
```python
import kaggle

# Download CIC-IDS-2017 dataset
kaggle.api.dataset_download_files('chethuhn/network-intrusion-dataset', path='data/', unzip=True)

# Download UNSW-NB15 dataset
kaggle.api.dataset_download_files('dhoogla/unswnb15', path='data/', unzip=True)

# Download CSIC-2010 Web Application Attacks dataset
kaggle.api.dataset_download_files('ispangler/csic-2010-web-application-attacks', path='data/', unzip=True)

# Download CSE-CIC-IDS2018 dataset
kaggle.api.dataset_download_files('solarmainframe/ids-intrusion-csv', path='data/', unzip=True)

print("Datasets downloaded successfully!")
```

$$✅ Successfully-done $$

# First Attempt of (Cleaning, Validating and Labeling):

## Cleaning of Data (Failed Attempt):
```python
import os
import pandas as pd

# Set data directory
data_dir = "data/"
cleaned_dir = "data/cleaned/"

# Ensure cleaned directory exists
os.makedirs(cleaned_dir, exist_ok=True)

# List all CSV files
csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

# Process each file in chunks instead of loading everything at once
chunk_size = 50000  # Adjust based on available RAM
for file in csv_files:
    file_path = os.path.join(data_dir, file)
    cleaned_path = os.path.join(cleaned_dir, file)
    
    print(f"\n🚀 Processing {file} in chunks...")
    
    # Open CSV & process in chunks
    with pd.read_csv(file_path, chunksize=chunk_size, low_memory=True) as reader:
        for i, chunk in enumerate(reader):
            # Drop missing values in the chunk
            chunk.dropna(inplace=True)
            
            # Save the first chunk with headers, append rest without headers
            mode = 'w' if i == 0 else 'a'
            header = True if i == 0 else False
            chunk.to_csv(cleaned_path, mode=mode, index=False, header=header)

            print(f"✅ Processed chunk {i+1} of {file}")

    print(f"📂 Cleaned file saved: {cleaned_path}")

print("\n🎯 All datasets cleaned & saved successfully!")
```

## Validating Data Labels and Cleaning (Failed Attempt):
```python
import os
import pandas as pd

# Set cleaned data directory
cleaned_dir = "data/cleaned/"

# List all CSV files
csv_files = [f for f in os.listdir(cleaned_dir) if f.endswith(".csv")]

# Check each dataset
for file in csv_files:
    file_path = os.path.join(cleaned_dir, file)
    
    try:
        # Load dataset
        df = pd.read_csv(file_path, nrows=5000)  # Load only first 5000 rows for speed
        print(f"\n📂 Checking {file} - {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Show first few rows
        print(df.head())

        # Check for label column
        possible_labels = ["attack", "attack_type", "label", "category", "class"]
        for col in df.columns:
            if any(keyword in col.lower() for keyword in possible_labels):
                print(f"✅ {file} is labeled! Label column found: {col}")
                break
        else:
            print(f"❌ {file} does NOT have a label column.")

    except Exception as e:
        print(f"⚠️ Error reading {file}: {e}")
```

## Processed Data Set Uploaded on the Kaggle:
##### -> Total 8.2Gbs Data was processed 
https://www.kaggle.com/datasets/ogguy11/apt-detection
https://www.kaggle.com/datasets/ogguy11/apt-detection-2

## Merging the Data (Failed Attempt):
 The Data merging got failed due to the uncleaned data set. Hence, conclusion was "The attempt to clean the data was also a failed attempt!"
 
https://www.kaggle.com/datasets/mulaimboy/mergeddata-aptdetection



## Validating the Data from Kaggle again(Outlier Checking):

```python
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
```

***-> The Results showed that data was not cleaned properly and the script did not handle the outliers efficiently***

***"Hence, the team restarted the cleaning procedure with the different python scripts"***

# Restarting the Process:

***-> Finding the Efficient Python Script "Clean.py" for Data Cleaning meanwhile validating the results with the "IsCleaned.py" Script***
## Failed Scripts:
```python
#Clean.py
import os
import pandas as pd

def clean_dataset(file_path, output_path):
    # Load the dataset
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        print("Error reading the file:", e)
        return

    print("Original Dataset Shape:", data.shape)

    # 1. Remove duplicate rows
    initial_rows = data.shape[0]
    data.drop_duplicates(inplace=True)
    print(f"Removed {initial_rows - data.shape[0]} duplicate rows.")

    # 2. Handle missing values
    # Drop columns with more than 50% missing values
    threshold = len(data) * 0.5
    cols_before = data.shape[1]
    data = data.dropna(thresh=threshold, axis=1)
    print(f"Dropped {cols_before - data.shape[1]} columns with >50% missing values.")

    # Fill missing values in numeric columns with the median
    numeric_cols = data.select_dtypes(include='number').columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

    # Fill missing values in categorical columns with the mode
    categorical_cols = data.select_dtypes(include='object').columns
    for col in categorical_cols:
        if not data[col].mode().empty:
            mode_value = data[col].mode()[0]
            data[col] = data[col].fillna(mode_value)

    # 3. Convert columns to appropriate data types
    # Normalize categorical data (e.g., strip whitespace, convert to lowercase)
    for col in categorical_cols:
        data[col] = data[col].str.strip().str.lower()

    # Check and convert columns with mixed types
    for col in data.columns:
        if data[col].dtype == 'object':
            try:
                data[col] = pd.to_numeric(data[col], errors='ignore')  # Convert to numeric if possible
            except Exception as e:
                print(f"Could not convert column {col}: {e}")

    # 4. Remove columns with zero variance (all values are the same)
    zero_variance_cols = [col for col in data.columns if data[col].nunique() <= 1]
    data.drop(columns=zero_variance_cols, inplace=True)
    print(f"Removed {len(zero_variance_cols)} zero-variance columns.")

    # 5. Handle inconsistent formatting in text columns
    for col in categorical_cols:
        # Replace common typos or inconsistencies (example: yes/no standardization)
        if data[col].dtype == 'object':
            data[col] = data[col].replace({'y': 'yes', 'n': 'no', ' ': None})

    # 6. Perform domain-specific validation
    # Example: Validate numeric ranges for specific columns
    if 'age' in data.columns:
        invalid_ages = data[(data['age'] < 0) | (data['age'] > 120)].shape[0]
        data = data[(data['age'] >= 0) & (data['age'] <= 120)]
        print(f"Removed {invalid_ages} rows with invalid ages.")

    # Example: Validate categorical columns against known categories
    if 'gender' in data.columns:
        valid_genders = ['male', 'female', 'other']
        invalid_genders = data[~data['gender'].isin(valid_genders)].shape[0]
        data = data[data['gender'].isin(valid_genders)]
        print(f"Removed {invalid_genders} rows with invalid genders.")

    # 7. Save the cleaned dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to: {output_path}")

# Example usage
file_path = "Cleaned Dataset/02-14-2018.csv"  # Replace with the path to your dataset
output_path = "Recleaned Dataset/02-14-2018.csv"  # Replace with the desired output file path
clean_dataset(file_path, output_path)
```
------------------------------------------------------------------------
```python
#isCleaned.py
import pandas as pd

def test_cleaned_data(file_path):
    # Load the dataset
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        print("Error reading the file:", e)
        return

    print("Testing Cleaned Dataset...")
    print("Dataset Shape:", data.shape)
    issues_found = False

    # 1. Check for missing values
    missing_count = data.isnull().sum().sum()
    if missing_count > 0:
        print(f"❌ Missing Values Detected: {missing_count}")
        print(data.isnull().sum())
        issues_found = True
    else:
        print("✅ No Missing Values.")

    # 2. Check for duplicate rows
    duplicate_count = data.duplicated().sum()
    if duplicate_count > 0:
        print(f"❌ Duplicate Rows Detected: {duplicate_count}")
        issues_found = True
    else:
        print("✅ No Duplicate Rows.")

    # 3. Check for zero-variance columns
    zero_variance_cols = [col for col in data.columns if data[col].nunique() <= 1]
    if zero_variance_cols:
        print(f"❌ Zero-Variance Columns Detected: {len(zero_variance_cols)}")
        print("Columns:", zero_variance_cols)
        issues_found = True
    else:
        print("✅ No Zero-Variance Columns.")

    # 4. Check for inconsistent formatting in text columns
    categorical_cols = data.select_dtypes(include='object').columns
    for col in categorical_cols:
        if data[col].str.contains(r"^\s+|\s+$", regex=True).any():
            print(f"❌ Inconsistent Formatting Detected in Column: {col}")
            issues_found = True
    print("✅ Text Columns Formatting Looks Consistent.")

    # 5. Validate column data types
    for col in data.columns:
        if data[col].dtype == 'object':
            try:
                pd.to_numeric(data[col], errors='raise')  # Validate if numeric data is stored as object
            except ValueError:
                print(f"❌ Column '{col}' contains non-numeric values but should be numeric.")
                issues_found = True
    print("✅ Column Data Types Look Valid.")

    # 6. Perform domain-specific validation
    if 'age' in data.columns:
        invalid_ages = data[(data['age'] < 0) | (data['age'] > 120)].shape[0]
        if invalid_ages > 0:
            print(f"❌ Invalid Ages Detected: {invalid_ages}")
            issues_found = True
        else:
            print("✅ All Ages Are Valid.")

    if 'gender' in data.columns:
        valid_genders = ['male', 'female', 'other']
        invalid_genders = data[~data['gender'].isin(valid_genders)].shape[0]
        if invalid_genders > 0:
            print(f"❌ Invalid Genders Detected: {invalid_genders}")
            issues_found = True
        else:
            print("✅ All Genders Are Valid.")

    # Final status
    if not issues_found:
        print("🎉 The dataset appears to be fully cleaned!")
    else:
        print("⚠️ Cleaning Issues Found. Please address the above.")

# Example usage
file_path = "Recleaned Dataset/02-14-2018.csv"  # Replace with your cleaned dataset path
test_cleaned_data(file_path)
```

***-> Using this clean script we tried to clean the data again and handle the outliers too.***
***-> But the isCleaned.py did not validate the results and validation report failed.***

## Final Scripts:

```python
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
INPUT_FILE = r"C:\Users\user\.cache\kagglehub\datasets\ogguy11\apt-detection\versions\1\02-23-2018.csv"  # 🟢 CHANGE INPUT PATH HERE
OUTPUT_DIR = r"D:\4th semester\SE\project\final_final"               # 🟢 CHANGE OUTPUT DIRECTORY HERE
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
cleaned_data.to_csv(os.path.join(OUTPUT_DIR, '2-23-2018_final_final.csv'), index=False)
outliers_df.to_csv(os.path.join(OUTPUT_DIR, 'all_outliers23.csv'), index=False)

if label_present:
    attack_outliers.to_csv(os.path.join(OUTPUT_DIR, 'attack_outliers23.csv'), index=False)
    benign_outliers.to_csv(os.path.join(OUTPUT_DIR, 'benign_outliers23.csv'), index=False)

# ========== STEP 7: VISUALIZATION ==========
print("\n🔄 [7/7] Generating visualizations...")
plt.figure(figsize=(12, 6))
sns.boxplot(data=data.select_dtypes(include=np.number).iloc[:, :10])
plt.title('Feature Distribution Before Cleaning')
plt.xticks(rotation=45)
plt.savefig(os.path.join(OUTPUT_DIR, 'pre_clean23.png'), bbox_inches='tight')

plt.figure(figsize=(12, 6))
sns.boxplot(data=cleaned_data.select_dtypes(include=np.number).iloc[:, :10])
plt.title('Feature Distribution After Cleaning')
plt.xticks(rotation=45)
plt.savefig(os.path.join(OUTPUT_DIR, 'post_clean23.png'), bbox_inches='tight')

print("\n✅ Cleaning complete! Results saved to:", OUTPUT_DIR)
print("📊 Visualizations saved as 'pre_clean.png' and 'post_clean.png'")
```
------------------------------------------------------------------------
```python
import pandas as pd
import numpy as np
import os

def validate_cleaning(original_path, cleaned_path, output_dir):
    # Check if the cleaned file exists
    if not os.path.exists(cleaned_path):
        print(f"❌ Cleaned file not found: {cleaned_path}")
        return
    
    # Load datasets
    original = pd.read_csv(original_path)
    cleaned = pd.read_csv(cleaned_path)
    
    print("\n🔍 Validation Report")
    print("====================")
    
    # 1. Check numeric conversion
    non_numeric_original = original.select_dtypes(exclude=[np.number]).columns
    non_numeric_cleaned = cleaned.select_dtypes(exclude=[np.number]).columns
    
    print("\n✅ Numeric Conversion Check:")
    print(f"Original non-numeric columns: {list(non_numeric_original)}")
    print(f"Cleaned non-numeric columns: {list(non_numeric_cleaned)}")
    
    # 2. Missing values check
    print("\n✅ Missing Values Check:")
    print(f"Original NaN count: {original.isna().sum().sum()}")
    print(f"Cleaned NaN count: {cleaned.isna().sum().sum()}")
    
    # 3. Infinite values check
    print("\n✅ Infinite Values Check:")
    print(f"Original inf count: {np.isinf(original.select_dtypes(include=[np.number])).sum().sum()}")
    print(f"Cleaned inf count: {np.isinf(cleaned.select_dtypes(include=[np.number])).sum().sum()}")
    
    # 4. Value range check
    print("\n✅ Value Range Check:")
    clipped_values = cleaned.select_dtypes(include=[np.number]).apply(
        lambda x: np.sum((x < -1e9) | (x > 1e9))).sum()
    print(f"Values outside [-1e9, 1e9] range: {clipped_values}")
    
    # 5. Outlier validation
    print("\n✅ Outlier Validation:")
    outliers = pd.read_csv(os.path.join(output_dir, "all_outliers.csv"))
    expected_removed = len(original) - len(cleaned)
    
    # Use set operations to determine the number of removed rows
    original_indices = set(original.index)
    cleaned_indices = set(cleaned.index)
    outlier_indices = set(outliers.index)
    
    actual_removed = len(outlier_indices - cleaned_indices)
    
    print(f"Expected removed rows: {expected_removed}")
    print(f"Actual removed rows: {actual_removed}")
    matching_percentage = actual_removed / expected_removed if expected_removed != 0 else 0
    print(f"Matching percentage: {matching_percentage:.2%}")

    # 6. Benign outlier retention check
    if 'Label' in cleaned.columns:
        benign_outliers = pd.read_csv(os.path.join(output_dir, "benign_outliers.csv"))
        retained_benign = cleaned.merge(benign_outliers, how='inner')
        print("\n✅ Benign Outlier Retention:")
        print(f"Total benign outliers: {len(benign_outliers)}")
        print(f"Retained benign outliers: {len(retained_benign)}")
        print(f"Retention percentage: {len(retained_benign)/len(benign_outliers):.2%}")

    # Final validation summary
    print("\n🔎 Validation Summary:")
    if (len(non_numeric_cleaned) == 0 and
        cleaned.isna().sum().sum() == 0 and
        clipped_values == 0 and
        (abs(actual_removed - expected_removed) < 5 or matching_percentage >= 0.80)):
        print("🟢 ALL CHECKS PASSED - Data cleaned successfully!")
    else:
        print("🔴 VALIDATION FAILED - Check cleaning steps")

# ========== CONFIGURATION ==========
ORIGINAL_FILE = r"Cleaned Dataset\02-14-2018.csv"  # Same as main script
OUTPUT_DIR = r"Recleaned Dataset\02-14-2018"                  # Same as main script

if __name__ == "__main__":
    validate_cleaning(ORIGINAL_FILE, os.path.join(OUTPUT_DIR, "cleaned_data.csv"), OUTPUT_DIR)
```

## APT 2 Problems:
***-> It had the largest File of 4.2Gb and we tried to clean it and validate it***
```python
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore

# ========== INTEL OPTIMIZATION ==========
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    print("Intel(R) Extension for Scikit-learn* enabled")
except ImportError:
    print("Intel Extension for Scikit-learn not available, using standard sklearn")

# ========== CONFIGURATION ==========
INPUT_FILE = "/home/kay/Documents/Workspace-S25/SE/SeProject/Data/dataset.csv"
OUTPUT_DIR = "/home/kay/Documents/Workspace-S25/SE/SeProject/CleanedData"
CHUNK_SIZE = 10000  # Smaller chunk size to reduce memory usage
TEMP_DIR = os.path.join(OUTPUT_DIR, "temp")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

print(f"\n🔧 Configuration:")
print(f"Input: {INPUT_FILE}")
print(f"Output: {OUTPUT_DIR}")
print(f"Chunk size: {CHUNK_SIZE}\n")

# ========== STEP 1: DATA LOADING & PREPROCESSING (CHUNK BY CHUNK) ==========
print("🔄 [1/5] Processing data in chunks...")

# Initialize counters and metadata collectors
total_rows = 0
total_outliers = 0
all_columns = None
has_label = False
chunks_processed = 0
temp_files = []

# Process each chunk
for chunk_num, chunk in enumerate(pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, low_memory=False)):
    chunk_rows = len(chunk)
    total_rows += chunk_rows
    chunks_processed += 1
    
    print(f"\n--- Processing chunk {chunk_num+1} ({chunk_rows} rows) ---")
    
    # Check for label column in first chunk
    if chunk_num == 0:
        has_label = 'Label' in chunk.columns
        all_columns = chunk.columns
        print(f"✓ Dataset has {len(all_columns)} columns" + (" with Label column" if has_label else " without Label column"))
    
    # Save label if present
    if has_label:
        labels = chunk['Label'].copy()
        chunk = chunk.drop(columns=['Label'])
    
    # Convert to numeric (memory optimization)
    for col in chunk.columns:
        if chunk[col].dtype == 'object':
            chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
    
    # Basic cleaning
    chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
    chunk = chunk.dropna(axis=1, thresh=len(chunk)//2)
    
    # Fill missing values with column medians
    for col in chunk.columns:
        if chunk[col].isna().any():
            chunk[col] = chunk[col].fillna(chunk[col].median())
    
    # Clip extreme values
    chunk = chunk.clip(lower=-1e9, upper=1e9)
    
    # ===== OUTLIER DETECTION (memory optimized & fixed) =====
    numeric_cols = chunk.select_dtypes(include=np.number).columns.tolist()
    chunk_outlier_mask = pd.Series(False, index=chunk.index)
    
    # IQR method (fixed indexing)
    for col in numeric_cols:
        Q1 = chunk[col].quantile(0.25)
        Q3 = chunk[col].quantile(0.75)
        IQR = Q3 - Q1
        col_outliers = (chunk[col] < (Q1 - 1.5*IQR)) | (chunk[col] > (Q3 + 1.5*IQR))
        chunk_outlier_mask = chunk_outlier_mask | col_outliers
    
    # Z-score method (fixed indexing)
    for col in numeric_cols:
        try:
            z = np.abs(zscore(chunk[col]))
            col_outliers = pd.Series(False, index=chunk.index)
            col_outliers.loc[z > 3] = True
            chunk_outlier_mask = chunk_outlier_mask | col_outliers
        except:
            continue
    
    # Count outliers using the mask
    chunk_outlier_count = chunk_outlier_mask.sum()
    total_outliers += chunk_outlier_count
    print(f"✓ Detected {chunk_outlier_count} outliers ({chunk_outlier_count/chunk_rows:.2%})")
    
    # Add back labels
    if has_label:
        chunk['Label'] = labels
    
    # Save outliers (using boolean mask)
    if chunk_outlier_count > 0:
        outliers_df = chunk[chunk_outlier_mask]
        outlier_file = os.path.join(TEMP_DIR, f"outliers_{chunk_num}.csv")
        outliers_df.to_csv(outlier_file, index=False)
    
    # Save cleaned chunk (without outliers)
    if chunk_outlier_count > 0:
        cleaned_chunk = chunk[~chunk_outlier_mask]
    else:
        cleaned_chunk = chunk
    
    # Save the cleaned chunk
    clean_file = os.path.join(TEMP_DIR, f"clean_{chunk_num}.csv")
    cleaned_chunk.to_csv(clean_file, index=False)
    temp_files.append(clean_file)
    
    # Report progress
    if chunks_processed % 10 == 0:
        print(f"Progress: Processed {total_rows:,} rows so far...")

# ========== STEP 2: COMBINE CLEANED CHUNKS ==========
print("\n🔄 [2/5] Combining cleaned chunks...")

# Combine the first few chunks for visualization (memory-safe)
visualization_chunks = []
for i in range(min(5, chunks_processed)):
    try:
        viz_data = pd.read_csv(os.path.join(TEMP_DIR, f"clean_{i}.csv"))
        viz_sample = viz_data.select_dtypes(include=np.number).iloc[:, :10].sample(min(1000, len(viz_data)))
        visualization_chunks.append(viz_sample)
        if len(pd.concat(visualization_chunks)) > 5000:  # Safety check
            break
    except Exception as e:
        print(f"Warning: Couldn't load chunk {i} for visualization: {e}")
        continue

if visualization_chunks:
    visualization_sample = pd.concat(visualization_chunks)
    print(f"✓ Prepared visualization sample with {len(visualization_sample)} rows")
else:
    print("⚠️ Could not prepare visualization sample")
    visualization_sample = None

# ========== STEP 3: PROCESS OUTLIERS ==========
print("\n🔄 [3/5] Processing outliers...")
outlier_files = [f for f in os.listdir(TEMP_DIR) if f.startswith("outliers_")]

if outlier_files and has_label:
    # Sample approach - process in small batches
    attack_count = 0
    benign_count = 0
    
    for outlier_file in outlier_files[:min(10, len(outlier_files))]:  # Limit to first 10 files
        try:
            outlier_chunk = pd.read_csv(os.path.join(TEMP_DIR, outlier_file))
            if 'Label' in outlier_chunk.columns:
                attack_count += len(outlier_chunk[outlier_chunk.Label != 'Benign'])
                benign_count += len(outlier_chunk[outlier_chunk.Label == 'Benign'])
        except Exception as e:
            print(f"Warning: Couldn't process outlier file {outlier_file}: {e}")
            continue
    
    print(f"✓ Outlier classification sample:\n- Attack: {attack_count}\n- Benign: {benign_count}")

# ========== STEP 4: GENERATE FINAL OUTPUT ==========
print("\n🔄 [4/5] Generating final outputs...")

# Create a file with information about the chunks
with open(os.path.join(OUTPUT_DIR, 'processing_summary.txt'), 'w') as f:
    f.write(f"Data Processing Summary\n")
    f.write(f"=====================\n\n")
    f.write(f"Total rows processed: {total_rows:,}\n")
    f.write(f"Total outliers detected: {total_outliers:,} ({total_outliers/total_rows:.2%})\n")
    f.write(f"Temporary files created: {len(temp_files)}\n\n")
    f.write(f"To combine all cleaned files:\n")
    f.write(f"  > import pandas as pd\n")
    f.write(f"  > import glob, os\n")
    f.write(f"  > all_files = glob.glob(os.path.join('{TEMP_DIR}', 'clean_*.csv'))\n")
    f.write(f"  > combined = pd.concat([pd.read_csv(f) for f in all_files])\n")
    f.write(f"  > combined.to_csv('{os.path.join(OUTPUT_DIR, 'cleaned_data.csv')}', index=False)\n")

print(f"✓ Created processing summary")

# ========== STEP 5: VISUALIZATION ==========
print("\n🔄 [5/5] Generating visualizations...")

try:
    if visualization_sample is not None and not visualization_sample.empty:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=visualization_sample)
        plt.title('Feature Distribution (Sample of Cleaned Data)')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(OUTPUT_DIR, 'data_distribution.png'), bbox_inches='tight')
        print("✓ Created distribution visualization")
    else:
        print("⚠️ Skipping visualization: no sample data available")
except Exception as e:
    print(f"⚠️ Could not create visualization: {e}")

print("\n✅ Memory-safe processing complete!")
print(f"📊 Results saved to: {OUTPUT_DIR}")
print(f"📋 Temporary files saved to: {TEMP_DIR}")
print("\nNext steps:")
print("1. Review the processing_summary.txt file")
print("2. Use the provided code to combine cleaned chunks if needed")
print("3. Or process each cleaned chunk individually for further analysis")
```

```python
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from tqdm import tqdm

def validate_cleaning(original_path, cleaned_path, output_dir):
    """
    Validate the data cleaning process by comparing original and cleaned datasets.
    
    Args:
        original_path (str): Path to the original CSV file
        cleaned_path (str): Path to the cleaned CSV file
        output_dir (str): Directory where outliers and temp files are stored
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if files exist
    if not os.path.exists(original_path):
        print(f"❌ Original file not found: {original_path}")
        return
    
    if not os.path.exists(cleaned_path):
        print(f"❌ Cleaned file not found: {cleaned_path}")
        return
    
    print(f"\n🔄 Loading datasets...")
    print(f"  - Original: {original_path}")
    print(f"  - Cleaned: {cleaned_path}")
    
    # Load datasets
    try:
        # Load in chunks if files are large
        original_chunks = pd.read_csv(original_path, chunksize=100000)
        original = pd.concat([chunk for chunk in tqdm(original_chunks, desc="Loading original")])
        
        cleaned = pd.read_csv(cleaned_path)
        print(f"  ✅ Loaded successfully")
    except Exception as e:
        print(f"  ❌ Error loading datasets: {e}")
        return
    
    print("\n🔍 Validation Report")
    print("====================")
    
    # 1. Basic comparison
    print("\n✅ Basic Comparison:")
    print(f"  Original shape: {original.shape}")
    print(f"  Cleaned shape: {cleaned.shape}")
    print(f"  Rows removed: {len(original) - len(cleaned)} ({(len(original) - len(cleaned))/len(original):.2%})")
    
    # 2. Check numeric conversion
    non_numeric_original = original.select_dtypes(exclude=[np.number]).columns.tolist()
    non_numeric_cleaned = cleaned.select_dtypes(exclude=[np.number]).columns.tolist()
    
    print("\n✅ Numeric Conversion Check:")
    print(f"  Original non-numeric columns: {non_numeric_original}")
    print(f"  Cleaned non-numeric columns: {non_numeric_cleaned}")
    
    # 3. Missing values check
    print("\n✅ Missing Values Check:")
    original_na_count = original.isna().sum().sum()
    cleaned_na_count = cleaned.isna().sum().sum()
    
    print(f"  Original NaN count: {original_na_count}")
    print(f"  Cleaned NaN count: {cleaned_na_count}")
    
    if cleaned_na_count == 0:
        print("  ✅ All missing values have been addressed")
    else:
        print("  ⚠️ Some missing values remain in the cleaned dataset")
    
    # 4. Infinite values check
    print("\n✅ Infinite Values Check:")
    try:
        original_inf_count = np.isinf(original.select_dtypes(include=[np.number])).sum().sum()
        cleaned_inf_count = np.isinf(cleaned.select_dtypes(include=[np.number])).sum().sum()
        
        print(f"  Original inf count: {original_inf_count}")
        print(f"  Cleaned inf count: {cleaned_inf_count}")
        
        if cleaned_inf_count == 0:
            print("  ✅ All infinite values have been addressed")
        else:
            print("  ⚠️ Some infinite values remain in the cleaned dataset")
    except Exception as e:
        print(f"  ⚠️ Could not check for infinite values: {e}")
    
    # 5. Value range check
    print("\n✅ Value Range Check:")
    try:
        clipped_values = cleaned.select_dtypes(include=[np.number]).apply(
            lambda x: np.sum((x < -1e9) | (x > 1e9))).sum()
        
        print(f"  Values outside [-1e9, 1e9] range: {clipped_values}")
        
        if clipped_values == 0:
            print("  ✅ All values are within the expected range")
        else:
            print("  ⚠️ Some values are outside the expected range")
    except Exception as e:
        print(f"  ⚠️ Could not check value ranges: {e}")
    
    # 6. Check for outliers in temp directory
    print("\n✅ Outlier Validation:")
    temp_dir = os.path.join(output_dir, "temp")
    outlier_files = glob(os.path.join(temp_dir, "outliers_*.csv"))
    
    if outlier_files:
        print(f"  Found {len(outlier_files)} outlier files")
        
        # Sample approach - load a sample of outlier files
        outlier_count = 0
        sample_size = min(10, len(outlier_files))
        
        for outlier_file in outlier_files[:sample_size]:
            try:
                outlier_chunk = pd.read_csv(outlier_file)
                outlier_count += len(outlier_chunk)
            except Exception as e:
                print(f"  ⚠️ Could not load outlier file {outlier_file}: {e}")
        
        print(f"  Sampled outlier count: {outlier_count} (from {sample_size} files)")
        
        # If we have Label column, check distribution
        if 'Label' in original.columns and 'Label' in cleaned.columns:
            print("\n✅ Label Distribution Check:")
            original_labels = original['Label'].value_counts(normalize=True)
            cleaned_labels = cleaned['Label'].value_counts(normalize=True)
            
            print("  Original dataset label distribution:")
            for label, pct in original_labels.items():
                print(f"    - {label}: {pct:.2%}")
                
            print("  Cleaned dataset label distribution:")
            for label, pct in cleaned_labels.items():
                print(f"    - {label}: {pct:.2%}")
    else:
        print("  ⚠️ No outlier files found for validation")
    
    # 7. Column preservation check
    print("\n✅ Column Preservation Check:")
    original_cols = set(original.columns)
    cleaned_cols = set(cleaned.columns)
    
    missing_cols = original_cols - cleaned_cols
    new_cols = cleaned_cols - original_cols
    
    if len(missing_cols) > 0:
        print(f"  ⚠️ Columns removed during cleaning: {missing_cols}")
    else:
        print("  ✅ All original columns preserved")
        
    if len(new_cols) > 0:
        print(f"  ℹ️ New columns added during cleaning: {new_cols}")
    
    # 8. Generate visualizations
    print("\n✅ Generating Visualizations:")
    try:
        # Create a directory for visualizations
        viz_dir = os.path.join(output_dir, "validation_visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Select numeric columns for visualization (limit to 10 for readability)
        numeric_cols = cleaned.select_dtypes(include=[np.number]).columns[:10].tolist()
        
        if numeric_cols:
            # Distribution comparison
            plt.figure(figsize=(15, 10))
            
            for i, col in enumerate(numeric_cols[:5], 1):
                plt.subplot(2, 3, i)
                
                # Sample data for performance
                orig_sample = original[col].dropna().sample(min(1000, len(original)))
                clean_sample = cleaned[col].dropna().sample(min(1000, len(cleaned)))
                
                sns.histplot(orig_sample, color='blue', alpha=0.5, label='Original')
                sns.histplot(clean_sample, color='red', alpha=0.5, label='Cleaned')
                
                plt.title(f'Distribution: {col}')
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'distribution_comparison.png'))
            print(f"  ✅ Created distribution comparison visualization")
            
            # Box plot comparison
            plt.figure(figsize=(15, 10))
            
            # Sample data for performance
            orig_sample = original[numeric_cols].dropna().sample(min(1000, len(original)))
            clean_sample = cleaned[numeric_cols].dropna().sample(min(1000, len(cleaned)))
            
            # Prepare data for boxplot
            orig_melted = pd.melt(orig_sample.reset_index(), id_vars=['index'], value_vars=numeric_cols)
            orig_melted['dataset'] = 'Original'
            
            clean_melted = pd.melt(clean_sample.reset_index(), id_vars=['index'], value_vars=numeric_cols)
            clean_melted['dataset'] = 'Cleaned'
            
            combined = pd.concat([orig_melted, clean_melted])
            
            # Create boxplot
            sns.boxplot(x='variable', y='value', hue='dataset', data=combined)
            plt.title('Feature Distribution Comparison')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'boxplot_comparison.png'))
            print(f"  ✅ Created boxplot comparison visualization")
        else:
            print("  ⚠️ No numeric columns available for visualization")
    except Exception as e:
        print(f"  ⚠️ Could not create visualizations: {e}")
    
    # 9. Final validation summary
    print("\n🔎 Final Validation Summary:")
    
    # Define validation checks
    checks = [
        ("Missing values addressed", cleaned_na_count == 0),
        ("Infinite values addressed", 'cleaned_inf_count' in locals() and cleaned_inf_count == 0),
        ("Values within expected range", 'clipped_values' in locals() and clipped_values == 0),
        ("All columns preserved", len(missing_cols) == 0 or 'Label' in missing_cols)
    ]
    
    # Count passed checks
    passed = sum(result for _, result in checks if result is not None)
    total = sum(1 for _, result in checks if result is not None)
    
    # Print results
    for check, result in checks:
        if result is None:
            print(f"  ⚠️ {check}: Could not verify")
        elif result:
            print(f"  ✅ {check}: Passed")
        else:
            print(f"  ❌ {check}: Failed")
    
    # Overall result
    if passed == total:
        print("\n🟢 ALL CHECKS PASSED - Data cleaned successfully!")
    else:
        print(f"\n🟡 PARTIAL VALIDATION - {passed}/{total} checks passed")
        print("    Review the report to identify potential issues")
    
    # Save report to file
    report_path = os.path.join(output_dir, "validation_report.txt")
    with open(report_path, 'w') as f:
        f.write("Data Cleaning Validation Report\n")
        f.write("==============================\n\n")
        f.write(f"Original file: {original_path}\n")
        f.write(f"Cleaned file: {cleaned_path}\n\n")
        f.write(f"Original shape: {original.shape}\n")
        f.write(f"Cleaned shape: {cleaned.shape}\n")
        f.write(f"Rows removed: {len(original) - len(cleaned)} ({(len(original) - len(cleaned))/len(original):.2%})\n\n")
        f.write(f"Validation result: {passed}/{total} checks passed\n")
    
    print(f"\n📋 Validation report saved to: {report_path}")
    print(f"📊 Visualizations saved to: {viz_dir}")

# ========== CONFIGURATION ==========
if __name__ == "__main__":
    # Set the file paths
    ORIGINAL_FILE = "/home/kay/Documents/Workspace-S25/SE/SeProject/Data/dataset.csv"
    CLEANED_FILE = "/home/kay/Documents/Workspace-S25/SE/SeProject/CleanedData/cleaned_data.csv"
    OUTPUT_DIR = "/home/kay/Documents/Workspace-S25/SE/SeProject/CleanedData"
    
    # Run validation
    validate_cleaning(ORIGINAL_FILE, CLEANED_FILE, OUTPUT_DIR)
```

# Cleaned Dataset on the Kaggle:
https://www.kaggle.com/datasets/a6ca3235119ffc39be6d0fa1561e055dcd9a92866f71f469b45495e1c57bb19d

------------------------------------------------------------------------
