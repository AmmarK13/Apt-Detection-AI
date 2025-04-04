import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
file_path = "/home/kay/Documents/Workspace-S25/SE/SeProject/APT DETECTION/z.after mids progress/dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX_CLEANED.csv"
df = pd.read_csv(file_path)

# ------------------------------
# 1. Enhanced Data Preparation
# ------------------------------
def prepare_data(data):
    """Handle zeros and negative values before transformations"""
    num_cols = data.select_dtypes(include="number").columns.tolist()
    
    # Identify skewed features dynamically
    skewed_features = [
        "Flow Bytes/s", "Flow Packets/s", "Bwd Packets/s",
        "Flow Duration", "Total Length of Fwd Packets"
    ]
    skewed_features = [f for f in skewed_features if f in data.columns]
    
    # Handle zeros/negatives for log transform
    if skewed_features:
        data[skewed_features] = data[skewed_features].apply(lambda x: x.where(x > 0, 1e-5))
        data[skewed_features] = np.log1p(data[skewed_features])
    
    return data

df = prepare_data(df)

# ------------------------------
# 2. Class-Balanced Outlier Removal
# ------------------------------
def balanced_outlier_removal(data, label_col="Label"):
    """Adaptive outlier handling with class-specific thresholds"""
    cleaned_data = pd.DataFrame()
    critical_features = [
        "SYN Flag Count", "URG Flag Count", 
        "Fwd Packet Length Max", "Bwd Packet Length Max"
    ]
    critical_features = [f for f in critical_features if f in data.columns]
    
    # Class-specific multipliers: Benign (0) gets lenient treatment
    multipliers = {0: 5, 1: 3}  # 0: benign, 1: attack
    
    for label in data[label_col].unique():
        subset = data[data[label_col] == label].copy()
        multiplier = multipliers.get(label, 3)
        
        # Select non-critical numerical features
        num_cols = subset.select_dtypes(include="number").columns.drop(label_col, errors="ignore")
        num_cols = [col for col in num_cols if col not in critical_features]
        
        if len(num_cols) == 0:
            cleaned_data = pd.concat([cleaned_data, subset])
            continue
            
        # Adaptive IQR calculation
        Q1 = subset[num_cols].quantile(0.25)
        Q3 = subset[num_cols].quantile(0.75)
        IQR = Q3 - Q1
        
        lower = Q1 - (multiplier * IQR)
        upper = Q3 + (multiplier * IQR)
        
        # Mask for non-critical features only
        mask = ((subset[num_cols] >= lower) & (subset[num_cols] <= upper)).all(axis=1)
        cleaned_data = pd.concat([cleaned_data, subset[mask]])
    
    return cleaned_data

df_clean = balanced_outlier_removal(df)

# ------------------------------
# 3. Intelligent Feature Capping
# ------------------------------
def safe_feature_capping(data):
    """Preserve attack patterns while controlling extremes"""
    cap_config = {
        "SYN Flag Count": (0.001, 0.999),
        "URG Flag Count": (0.0, 0.99),
        "Fwd Packet Length Max": (0.01, 0.99),
        "Bwd Packet Length Max": (0.01, 0.99)
    }
    
    for feature, (low_q, high_q) in cap_config.items():
        if feature in data.columns:
            low = data[feature].quantile(low_q)
            high = data[feature].quantile(high_q)
            data[feature] = np.clip(data[feature], low, high)
    
    return data

df_clean = safe_feature_capping(df_clean)

# ------------------------------
# 4. Validation & Output
# ------------------------------
def analyze_results(original, cleaned):
    """Compare pre/post processing metrics"""
    print("\n=== Class Distribution ===")
    print("Original:\n", original["Label"].value_counts())
    print("\nCleaned:\n", cleaned["Label"].value_counts())
    
    print("\n=== Data Preservation ===")
    print(f"Total Samples: {len(original)} → {len(cleaned)} "
          f"({len(cleaned)/len(original)*100:.1f}% retained)")
    
    print("\n=== Feature Ranges ===")
    num_cols = cleaned.select_dtypes(include="number").columns
    for col in num_cols[:5]:  # Show first 5 features
        print(f"{col}: {cleaned[col].min():.2f} - {cleaned[col].max():.2f}")

analyze_results(df, df_clean)

# Save and visualize
output_path = file_path.replace(".csv", "_BALANCED_CLEAN.csv")
df_clean.to_csv(output_path, index=False)

plt.figure(figsize=(14, 6))
sns.boxplot(data=df_clean.select_dtypes(include="number"), orient="h", palette="Set2")
plt.title("Final Feature Distributions", fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()