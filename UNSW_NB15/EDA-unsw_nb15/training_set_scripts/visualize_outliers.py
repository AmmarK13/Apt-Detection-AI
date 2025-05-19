import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
file_path = "UNSW_NB15/cleaned_data/training_dups_removed.csv"  # Update if needed
df = pl.read_csv(file_path)

# Convert to Pandas for visualization
df_pandas = df.to_pandas()

# Select numerical columns
num_cols = ["dur", "sload", "dload", "spkts", "dpkts", "sinpkt", "dinpkt", "tcprtt", "synack", "ackdat"]

# Boxplots
plt.figure(figsize=(15, 8))
df_pandas[num_cols].boxplot(rot=45)
plt.title("Boxplot of Numerical Features (Checking Outliers)")
plt.show()

# Histograms
df_pandas[num_cols].hist(figsize=(15, 10), bins=50, layout=(3, 4))
plt.suptitle("Histograms of Numerical Features")
plt.show()


# Plot distributions for key numerical features
plt.figure(figsize=(15, 8))
for i, col in enumerate(["sload", "dload", "dur", "spkts", "dpkts"]):
    plt.subplot(2, 3, i + 1)
    sns.histplot(df_pandas[col], bins=50, kde=True)
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()