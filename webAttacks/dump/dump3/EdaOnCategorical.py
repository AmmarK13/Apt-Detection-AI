import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif

# Load the cleaned and preprocessed dataset
df = pd.read_csv("data/Cleaned_Preprocessed.csv")

# Define the target column
target_col = 'classification'

# ----------------------------------------
# 2. Pure Categorical Analysis
# ----------------------------------------
# (Assuming all columns except the target are categorical)
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

# Target Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=target_col, data=df)
plt.title("Distribution of Target (classification)")
plt.xlabel("Classification")
plt.ylabel("Count")
plt.show()  # Display the plot

if cat_cols:
    for col in cat_cols:
        plt.figure(figsize=(10, 4))
        # Get the order for the countplot
        order = df[col].value_counts().index.tolist()
        sns.countplot(data=df, y=col, order=order)
        plt.title(f"Count Plot for {col}")
        plt.xlabel("Count")
        plt.ylabel(col)
        plt.show()  # Display the plot
else:
    print("No categorical features to analyze in raw form.")
