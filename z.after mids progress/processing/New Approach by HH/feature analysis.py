import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re

# === Load your dataset ===
df = pd.read_csv("/home/kay/Documents/Workspace-S25/SE/SeProject/APT DETECTION/z.after mids progress/processing/New Approach by HH/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX_CLEANED_BALANCED_CLEAN.csv")

# === Set up output directory ===
output_dir = "feature_analysis_outputs"
os.makedirs(output_dir, exist_ok=True)

# === 1. Feature Distribution Analysis ===
numeric_cols = df.select_dtypes(include=np.number).columns

print(f"\n📊 Analyzing distributions for {len(numeric_cols)} numeric features...\n")

# Create folder to save plots
dist_dir = os.path.join(output_dir, "feature_distributions")
os.makedirs(dist_dir, exist_ok=True)

# Function to clean column names for safe file naming
def clean_filename(name):
    return re.sub(r'[^\w\-_.]', '_', name)

for idx, col in enumerate(numeric_cols, 1):  # start index at 1
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], bins=50, kde=True, color='teal')
    plt.title(f"Distribution of: {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()

    safe_col = clean_filename(col)
    filename = f"{idx}.{safe_col}.png"
    plt.savefig(os.path.join(dist_dir, filename))
    plt.close()
    print(f"📈 Saved distribution plot for: {col} as {filename}")

# === 2. Constant / Low Variance Feature Removal ===
low_variance_cols = [col for col in numeric_cols if df[col].nunique() <= 1]

print("\n🧹 Checking for constant or low variance features...")
if low_variance_cols:
    print(f"❌ Dropping columns with no variation: {low_variance_cols}")
    df.drop(columns=low_variance_cols, inplace=True)
else:
    print("✅ No constant or low variance features found.")

# === Save dataset after removing bad features ===
cleaned_path = "dataset_after_variance_check.csv"
df.to_csv(cleaned_path, index=False)
print(f"\n📁 Updated dataset saved to: {cleaned_path}")
