import pandas as pd
import numpy as np

# === Load your dataset ===
df = pd.read_csv("dataset_scaled.csv")  # change to your path

# === Select numeric columns (exclude Label if needed) ===
numeric_cols = df.select_dtypes(include=np.number).columns
numeric_cols = numeric_cols.drop("Label", errors="ignore")  # drop Label if present

print(f"\n🔎 Checking min-max range of {len(numeric_cols)} numeric features:\n")

scaling_needed = False

for col in numeric_cols:
    min_val = df[col].min()
    max_val = df[col].max()
    range_val = max_val - min_val

    print(f"{col:<30} ➜ Min: {min_val:.2f} | Max: {max_val:.2f} | Range: {range_val:.2f}")
    
    if range_val > 100 or max_val > 1e3:
        scaling_needed = True

# === Final Verdict ===
print("\n📊 Verdict:")
if scaling_needed:
    print("⚠️ Scaling is likely NEEDED due to large value ranges in some features.")
else:
    print("✅ Scaling may NOT be necessary. Feature ranges look okay.")
