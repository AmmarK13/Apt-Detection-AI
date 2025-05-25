import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif

# -------------------------------
# Load Datasets
# -------------------------------
# Raw (categorical) dataset
df_raw = pd.read_csv("data/Cleaned_Preprocessed.csv")
target_col = 'classification'

# Encoded (numerical) dataset (assumed to be created previously)
df_enc = pd.read_csv("data/Cleaned_Preprocessed_Numerical.csv")

# -------------------------------
# Analysis on Raw Data (Categorical)
# -------------------------------
print("=== RAW (Categorical) Data Analysis ===")

# 1. Class distribution of the target (raw)
print("\nTarget Distribution (Raw Data):")
target_distribution = df_raw[target_col].value_counts(normalize=True)
print(target_distribution)

# 2. Mutual Information for Categorical Features (using one-hot encoding temporarily)
cat_cols = df_raw.select_dtypes(include=['object']).columns.tolist()
if cat_cols:
    # Create one-hot encoded version for mutual information computation
    df_raw_ohe = pd.get_dummies(df_raw[cat_cols], drop_first=True)
    mi_raw = mutual_info_classif(df_raw_ohe, df_raw[target_col], random_state=42)
    mi_raw_series = pd.Series(mi_raw, index=df_raw_ohe.columns).sort_values(ascending=False)
    print("\nTop 10 Features by Mutual Information (Raw Data - One-hot encoded for MI):")
    print(mi_raw_series.head(10))
else:
    print("No categorical features found in raw data.")

# -------------------------------
# Analysis on Encoded Data (Numerical)
# -------------------------------
print("\n=== ENCODED (Numerical) Data Analysis ===")

# 1. Correlation Matrix and Correlations with the Target
corr_matrix = df_enc.corr()
print("\nCorrelation Matrix with target:")
correlations_enc = corr_matrix[target_col].sort_values(ascending=False)
print(correlations_enc)

# Flag potential leakage features (e.g., abs(correlation) > 0.8, excluding the target itself)
leakage_threshold = 0.8
leakage_features = correlations_enc[abs(correlations_enc) > leakage_threshold].index.tolist()
if target_col in leakage_features:
    leakage_features.remove(target_col)
if leakage_features:
    print("Potential leakage features in encoded data (|corr| > 0.8):", leakage_features)
else:
    print("No leakage features detected in encoded data based on correlation threshold.")

# 2. Mutual Information for Encoded Features
X_enc = df_enc.drop(target_col, axis=1)
y_enc = df_enc[target_col]
mi_enc = mutual_info_classif(X_enc, y_enc, random_state=42)
mi_enc_series = pd.Series(mi_enc, index=X_enc.columns).sort_values(ascending=False)
print("\nTop 10 Features by Mutual Information (Encoded Data):")
print(mi_enc_series.head(10))

# -------------------------------
# Comparison Summary
# -------------------------------
avg_mi_raw = mi_raw_series.mean() if cat_cols else None
avg_mi_enc = mi_enc_series.mean()
print("\nAverage Mutual Information (Raw Data - one-hot MI):", avg_mi_raw)
print("Average Mutual Information (Encoded Data):", avg_mi_enc)

# Optionally, you can also plot side-by-side comparisons.
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.barplot(x=mi_raw_series.head(10).values, y=mi_raw_series.head(10).index)
plt.title("Top MI Features (Raw Data)")
plt.xlabel("Mutual Information")

plt.subplot(1, 2, 2)
sns.barplot(x=mi_enc_series.head(10).values, y=mi_enc_series.head(10).index)
plt.title("Top MI Features (Encoded Data)")
plt.xlabel("Mutual Information")
plt.tight_layout()
plt.show()

print("\nComparison complete. Review the printed outputs and plots to decide if converting data to numerical form adds valuable insight or if the raw categorical analysis is sufficient for your needs.")
