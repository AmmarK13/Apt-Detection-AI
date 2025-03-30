import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
file_path = "UNSW_NB15/cleaned_data/reduced_training_set.csv"
df = pl.read_csv(file_path)

# Select only numeric columns for correlation
numeric_cols = df.select(pl.col(pl.Float64)).columns
df_numeric = df.select(numeric_cols)

# Compute correlation matrix
corr_matrix = df_numeric.to_pandas().corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.savefig("feature_correlation.png")
plt.show()