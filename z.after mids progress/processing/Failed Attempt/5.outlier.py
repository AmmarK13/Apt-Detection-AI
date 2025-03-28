import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load cleaned dataset
file_path = "/home/kay/Documents/Workspace-S25/SE/SeProject/APT DETECTION/z.after mids progress/dataset/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
df = pd.read_csv(file_path)

# Select only numerical columns
num_cols = df.select_dtypes(include=["number"]).columns

# Plot boxplots for outlier visualization
plt.figure(figsize=(12, 6))
df[num_cols].boxplot(rot=90, grid=False)
plt.title("Boxplot of Numerical Features (Before Outlier Handling)")
plt.show()


# -------------------- Outlier Detection Using IQR --------------------
# Compute Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df[num_cols].quantile(0.25)
Q3 = df[num_cols].quantile(0.75)
IQR = Q3 - Q1

# Define lower and upper bounds for outliers 
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)

outliers = ((df[num_cols]<lower_bound)| (df[num_cols]>upper_bound)).sum()
print("Number of Outliers per feature before handling :", outliers)

# -------------------- Handling Outliers --------------------
# Option 1: Remove outliers
df_no_outliers = df.copy()
for col in num_cols:
    df_no_outliers = df_no_outliers[(df_no_outliers[col] >= lower_bound[col]) & (df_no_outliers[col] <= upper_bound[col])]

print(f"Rows before removing outliers: {df.shape[0]}")
print(f"Rows after removing outliers: {df_no_outliers.shape[0]}")

# Plot boxplots after outlier removal
plt.figure(figsize=(12, 6))
df_no_outliers[num_cols].boxplot(rot=90, grid=False)
plt.title("Boxplot of Numerical Features (After Outlier Handling)")
plt.show()

# Save cleaned dataset
df_no_outliers.to_csv(file_path, index=False)
print(f"Cleaned dataset saved to {file_path}")
