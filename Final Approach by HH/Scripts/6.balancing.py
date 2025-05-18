import pandas as pd

# Load dataset
df = pd.read_csv("dataset_scaled.csv")

# Track columns to drop
columns_to_drop = []

# Iterate through columns (excluding 'Label')
for col in df.columns:
    if col == 'Label':
        continue

    unique_vals = df[col].nunique()
    if unique_vals < 10:
        value_counts = df[col].value_counts(normalize=True)
        top_percentage = value_counts.iloc[0] * 100

        # Drop if it's NOT between 45% and 55%
        if top_percentage < 45 or top_percentage > 55:
            columns_to_drop.append(col)

# Drop the columns
df.drop(columns=columns_to_drop, inplace=True)

# Save cleaned dataset
df.to_csv("dataset_scaled_cleaned.csv", index=False)

# Save dropped column names
with open("dropped_imbalanced_columns.txt", "w") as f:
    f.write("\n".join(columns_to_drop))

# Print summary
print("🧹 Dropped imbalanced low-uniqueness columns:")
for col in columns_to_drop:
    print(f"  ➤ {col}")

print("\n💾 Saved cleaned dataset as 'dataset_scaled_cleaned.csv'")
print("📝 Dropped column list saved to 'dropped_imbalanced_columns.txt'")
