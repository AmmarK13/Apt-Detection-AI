import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
file_path = "/home/kay/Documents/Workspace-S25/SE/SeProject/APT DETECTION/z.after mids progress/dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX_CLEANED.csv"
df = pd.read_csv(file_path)

# ------------------------------
# 1. Check and Update Feature Lists
# ------------------------------
# Get existing numerical columns
num_cols = df.select_dtypes(include="number").columns.tolist()

# Update skewed features list based on available columns
skewed_features = [
    "Flow Bytes/s", 
    "Flow Packets/s", 
    "Bwd Packets/s",
    "Flow Duration",
    "Total Length of Fwd Packets"
]

# Keep only features that actually exist in the dataset
skewed_features = [f for f in skewed_features if f in df.columns]

# ------------------------------
# 2. Log-Transform Skewed Features
# ------------------------------
if skewed_features:  # Only run if there are features to transform
    df[skewed_features] = df[skewed_features].apply(lambda x: np.log1p(x))

# ------------------------------
# 3. Visualize Outliers (Before Handling)
# ------------------------------
plt.figure(figsize=(15, 8))
sns.boxplot(data=df[num_cols], orient="h", palette="Set2")
plt.title("Outlier Distribution - Before Handling", fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ------------------------------
# 4. Stratified Outlier Removal (Updated)
# ------------------------------
def stratified_outlier_removal(data, label_col="Label", multiplier=3):
    cleaned_data = pd.DataFrame()
    
    # Updated critical features list based on available columns
    CRITICAL_FEATURES = [
        "SYN Flag Count", 
        "URG Flag Count",
        "Fwd Packet Length Max",
        "Bwd Packet Length Max",
        "Flow IAT Max"
    ]
    CRITICAL_FEATURES = [f for f in CRITICAL_FEATURES if f in data.columns]
    
    for label in data[label_col].unique():
        subset = data[data[label_col] == label].copy()
        subset_num_cols = subset.select_dtypes(include="number").columns.drop(label_col, errors="ignore")
        subset_num_cols = [col for col in subset_num_cols if col not in CRITICAL_FEATURES]
        
        if not subset_num_cols:  # Skip if no numerical columns left
            continue
            
        Q1 = subset[subset_num_cols].quantile(0.25)
        Q3 = subset[subset_num_cols].quantile(0.75)
        IQR = Q3 - Q1
        
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
        
        mask = ((subset[subset_num_cols] >= lower) & (subset[subset_num_cols] <= upper)).all(axis=1)
        cleaned_data = pd.concat([cleaned_data, subset[mask]])
    
    return cleaned_data

df_clean = stratified_outlier_removal(df, multiplier=3)

# ------------------------------
# 5. Cap Critical Features (Updated)
# ------------------------------
CRITICAL_TO_CAP = {
    "SYN Flag Count": (0.01, 0.99),
    "URG Flag Count": (0.0, 0.95),
    "Fwd Packet Length Max": (0.05, 0.95),
}

# Filter features that actually exist
CRITICAL_TO_CAP = {k: v for k, v in CRITICAL_TO_CAP.items() if k in df_clean.columns}

for feature, (low_q, high_q) in CRITICAL_TO_CAP.items():
    low = df_clean[feature].quantile(low_q)
    high = df_clean[feature].quantile(high_q)
    df_clean[feature] = df_clean[feature].clip(low, high)

# ------------------------------
# 6. Final Output & Validation
# ------------------------------
print("\nFinal Class Distribution:")
print(df_clean["Label"].value_counts())

print("\nShape Before Cleaning:", df.shape)
print("Shape After Cleaning:", df_clean.shape)

# Save processed data
output_path = file_path.replace(".csv", "_STRATIFIED_CLEAN.csv")
df_clean.to_csv(output_path, index=False)
print(f"\nSaved cleaned dataset to: {output_path}")

# Visualization
plt.figure(figsize=(14, 6))
sns.boxplot(data=df_clean.select_dtypes(include="number"), orient="h", palette="Set2")
plt.title("Feature Distributions After Cleaning", fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()