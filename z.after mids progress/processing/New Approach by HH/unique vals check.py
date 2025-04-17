import pandas as pd
import numpy as np

# === Load your dataset ===
df = pd.read_csv("dataset_balanced.csv")

# === Select numeric columns ===
numeric_cols = df.select_dtypes(include=np.number).columns

print("\n🔎 Checking unique value counts in numeric columns:\n")

for col in numeric_cols:
    unique_vals = df[col].nunique(dropna=False)
    print(f"{col}: {unique_vals} unique value(s)")

# === Identify truly constant/low variance columns ===
low_variance_cols = [col for col in numeric_cols if df[col].nunique(dropna=False) <= 1]

print("\n🧹 Result:")
if low_variance_cols:
    print(f"❌ Constant/Low variance columns found: {low_variance_cols}")
else:
    print("✅ All numeric columns have more than 1 unique value.")
