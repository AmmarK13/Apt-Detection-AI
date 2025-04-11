import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np

#load dataset
file_path = "/home/kay/Documents/Workspace-S25/SE/SeProject/APT DETECTION/z.after mids progress/dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
df = pd.read_csv(file_path)

# Ensure the 'Label' column is numeric (1 = Attack, 0 = Benign)
df["Label"] = df["Label"].astype(int)

# --- Step 1: Remove features highly correlated with Label (leakage) ---
# Compute correlations with Label
label_correlation = df.corr()["Label"].abs().sort_values(ascending=False)

# Set threshold for leakage (adjust as needed)
LEAKAGE_THRESHOLD = 0.5
leaky_features = label_correlation[label_correlation > LEAKAGE_THRESHOLD]

# Remove 'Label' from leaky_features
if "Label" in leaky_features.index:
    leaky_features = leaky_features.drop("Label")

print("\n--- Features Correlated with Label (Leakage Candidates) ---")
print(f"Threshold: > {LEAKAGE_THRESHOLD}")
print(leaky_features.to_string(header=False))  # Print features + correlation values

# Drop leaky features
df_cleaned = df.drop(columns=leaky_features.index.tolist())

# --- Step 2: Remove redundant features (correlated with each other) ---
# Compute feature-feature correlation matrix (exclude Label)
corr_matrix = df_cleaned.drop(columns=["Label"]).corr().abs()

# Find all feature pairs with correlation > 0.9
redundant_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        feature_i = corr_matrix.columns[i]
        feature_j = corr_matrix.columns[j]
        corr_value = corr_matrix.iloc[i, j]
        if corr_value > 0.9:
            redundant_pairs.append((feature_i, feature_j, corr_value))

# Print redundant pairs with correlation values
print("\n--- Redundant Feature Pairs (Correlation > 0.9) ---")
for pair in redundant_pairs:
    print(f"{pair[0]} <-> {pair[1]} : {pair[2]:.4f}")

# Identify columns to drop (from upper triangle)
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.9)]
# After identifying "to_drop", retain critical features
CRITICAL_FEATURES = ["SYN Flag Count", "ECE Flag Count", "Bwd IAT Max", "Fwd Packets/s"]
to_drop = [col for col in to_drop if col not in CRITICAL_FEATURES]

print(f"Final features to drop: {to_drop}")
df_cleaned = df_cleaned.drop(columns=to_drop)

# --- Save cleaned dataset ---
cleaned_file_path = file_path.replace(".csv", "_CLEANED.csv")
df_cleaned.to_csv(cleaned_file_path, index=False)
print(f"\nCleaned dataset saved to: {cleaned_file_path}")

# --- Diagnostics ---
print("\nOriginal columns:", len(df.columns))
print("Remaining columns:", len(df_cleaned.columns))