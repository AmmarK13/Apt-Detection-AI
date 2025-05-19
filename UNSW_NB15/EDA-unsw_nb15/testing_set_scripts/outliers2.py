# sanity check 1 and 2 for outliers
# check 1: distribution of data
# check 2: check and drop low variance feature

import matplotlib.pyplot as plt  
import seaborn as sns  
import pandas as pd
import numpy as np


# Load the cleaned training dataset
df = pd.read_csv("UNSW_NB15/cleaned_data/testing_outliers.csv")



df.hist(figsize=(15, 12), bins=50, edgecolor='black')  
plt.suptitle("Feature Distributions After Preprocessing", fontsize=3)  
plt.show()




# cgeck 2
# Select only numeric columns
df_numeric = df.select_dtypes(include=[np.number])

# Compute variance
variance = df_numeric.var()

# Identify low-variance features
low_variance_features = variance[variance < 1e-6]  # Features with extremely low variance

print("⚠️ Low variance features:\n", low_variance_features)

print("\nCheck the unique values in these columns")
for col in low_variance_features.index:
    print(f"{col}: {df[col].unique()}")

df_cleaned = df.drop(columns=low_variance_features.index)
print("\nDropped low-variance features.")

output_path = "UNSW_NB15/cleaned_data/testing_outliers_v2.csv"
df_cleaned.to_csv(output_path, index=False)
print(f"Processed dataset saved as '{output_path}'")



