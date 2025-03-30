import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("UNSW_NB15/cleaned_data/training_dups_removed.csv")

# Keep only numeric columns
df_numeric = df.select_dtypes(include=["number"])

# Compute correlation matrix
corr_matrix = df_numeric.corr()

# Plot heatmap
plt.figure(figsize=(12, 10))  # Adjusted figure size
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", annot_kws={"size": 6})
plt.title("Feature Correlation Heatmap")
plt.show()

# Print correlation with label
if "label" in df_numeric.columns:
    print("Correlation with 'label':")
    print(df_numeric.corr()["label"].sort_values(ascending=False))
else:
    print("Column 'label' not found in numeric columns.")

# Compute absolute correlation matrix
corr_matrix = df_numeric.corr().abs()

# Set correlation threshold
threshold = 0.98  # Features with correlation above this are considered redundant

# Find feature pairs with high correlation
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):  # Avoid duplicates
        if corr_matrix.iloc[i, j] > threshold:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))

# Print redundant feature pairs
if high_corr_pairs:
    print("Highly Correlated Feature Pairs (above 0.98):")
    for pair in high_corr_pairs:
        print(f"{pair[0]} ↔ {pair[1]}")
else:
    print("No highly correlated features found above the threshold.")



# DROP HIGH CORR FEATURES=============================================
# Drop redundant features
df_cleaned = df.drop(columns=["sloss", "dloss", "dwin", "is_ftp_login"])  # Keeping ct_ftp_cmd instead


# Display the new shape of the dataset
print(f"Original shape: {df.shape}")
print(f"New shape after dropping correlated features: {df_cleaned.shape}")


# Save the cleaned dataset
df_cleaned.to_csv("UNSW_NB15/cleaned_data/training_dataLeakage.csv", index=False)
print("Cleaned dataset saved as 'training_dataLeakage.csv'")