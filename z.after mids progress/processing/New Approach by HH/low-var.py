import pandas as pd
import numpy as np

# === Load your dataset ===
df = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX_CLEANED_BALANCED_CLEAN.csv")

# === Numeric columns (excluding 'Label') ===
numeric_cols = df.select_dtypes(include=np.number).columns
numeric_cols = numeric_cols.drop("Label", errors='ignore')

low_variance_cols = []

print("\n🧹 Checking for constant or imbalanced binary features...\n")

for col in numeric_cols:
    unique_vals = df[col].nunique(dropna=False)
    
    if unique_vals == 1:
        print(f"❌ {col} has only one unique value.")
        low_variance_cols.append(col)
    elif unique_vals == 2:
        counts = df[col].value_counts(normalize=True)
        max_ratio = counts.max()
        if max_ratio >= 0.99:
            print(f"⚠️ {col} has 2 values but is imbalanced ({max_ratio*100:.2f}% vs {100-max_ratio*100:.2f}%).")
            low_variance_cols.append(col)

if low_variance_cols:
    print(f"\n📉 Dropping low variance/imbalanced columns: {low_variance_cols}")
    df.drop(columns=low_variance_cols, inplace=True)
else:
    print("✅ No low variance or highly imbalanced binary features found.")

# === Save cleaned dataset ===
df.to_csv("dataset_after_low_variance_removed.csv", index=False)
print("\n📁 Final cleaned dataset saved to: dataset_after_low_variance_removed.csv")
