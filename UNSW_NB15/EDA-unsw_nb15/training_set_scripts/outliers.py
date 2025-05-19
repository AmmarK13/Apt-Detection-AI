import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from sklearn.preprocessing import MinMaxScaler

# Load the cleaned training dataset
df = pd.read_csv("UNSW_NB15/cleaned_data/training_dataLeakage.csv")

# Keep only numeric columns
df_numeric = df.select_dtypes(include=["number"])

# Compute Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df_numeric.quantile(0.25)
Q3 = df_numeric.quantile(0.75)

# Compute IQR
IQR = Q3 - Q1

# Define outlier thresholds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Outlier capping (Winsorization)
df_cleaned = df.copy()
for col in df_numeric.columns:
    df_cleaned[col] = df[col].clip(lower=lower_bound[col], upper=upper_bound[col])

print("Outliers capped at 1.5 * IQR range.")

# Additional Fixes

# 1. Apply log transformation to highly skewed features
log_features = ["sload", "dload", "rate", "sbytes", "dbytes"]
df_cleaned[log_features] = np.log1p(df_cleaned[log_features])  # log(1 + x) to handle zeros


# Features that had log transformation
log_features = ["sload", "dload", "rate", "sbytes", "dbytes"]
# Compute skewness before and after
skew_before = df[log_features].apply(skew)
skew_after = df_cleaned[log_features].apply(skew)
# Compare results
skew_comparison = pd.DataFrame({"\n\nbefore Log Transform": skew_before, "After Log Transform": skew_after})
print(skew_comparison)



# 2. Apply percentile-based capping for certain features
percentile_features = ["ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_ftp_cmd", "ct_flw_http_mthd"]
for col in percentile_features:
    lower = np.percentile(df[col], 1)  # 1st percentile
    upper = np.percentile(df[col], 99)  # 99th percentile
    df_cleaned[col] = df[col].clip(lower, upper)

# 3. Normalize all features using Min-Max Scaling
scaler = MinMaxScaler()
df_cleaned[df_numeric.columns] = scaler.fit_transform(df_cleaned[df_numeric.columns])

print("\nFinal transformations applied: Log scaling, percentile capping, and Min-Max scaling.")

# Compare min/max values before and after capping
for col in df_numeric.columns:
    before_min, before_max = df[col].min(), df[col].max()
    after_min, after_max = df_cleaned[col].min(), df_cleaned[col].max()
    
    if before_min != after_min or before_max != after_max:  # Check if capping changed anything
        print(f"{col}:")
        print(f"  Before capping: Min={before_min}, Max={before_max}")
        print(f"  After capping:  Min={after_min}, Max={after_max}\n")

# BOX PLOT - BEFORE AND AFTER =================================================
features_to_check = ["sbytes", "dbytes", "rate", "sload", "dload", "sinpkt", "tcprtt"]

plt.figure(figsize=(15, 10))

# Plot BEFORE capping
for i, feature in enumerate(features_to_check, 1):
    plt.subplot(2, 4, i)  # Arrange plots in a grid
    sns.boxplot(x=df[feature])  # Use ORIGINAL dataset before capping
    plt.title(f"Before Capping: {feature}")

plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 10))

# Plot AFTER capping
for i, feature in enumerate(features_to_check, 1):
    plt.subplot(2, 4, i)  # Arrange plots in a grid
    sns.boxplot(x=df_cleaned[feature])
    plt.title(f"After Capping: {feature}")

plt.tight_layout()
plt.show()

# Save the processed dataset
output_path = "UNSW_NB15/cleaned_data/training_outliers.csv"
df_cleaned.to_csv(output_path, index=False)
print(f"Processed dataset saved as '{output_path}'")
