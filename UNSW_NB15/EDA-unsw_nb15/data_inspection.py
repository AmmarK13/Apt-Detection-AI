import polars as pl


print("----training file----")

# Load the dataset
df = pl.read_csv("data/UNSW_NB15_training-set.csv")  

# Check number of rows and columns
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# Check for missing values
missing_vals = df.null_count()
print("Missing Values:\n", missing_vals)

# Check summary statistics (to detect outliers)
print(df.describe())

# Check unique values in categorical columns (detect anomalies)
for col in ["service", "state", "attack_cat"]:
    print(f"Unique values in {col}: {df[col].unique()}")


# ==============================================================================================


print("----testing file----")
# Load the dataset
df = pl.read_csv("data/UNSW_NB15_testing-set.csv")  

# Check number of rows and columns
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# Check for missing values
missing_vals = df.null_count()
print("Missing Values:\n", missing_vals)

# Check summary statistics (to detect outliers)
print(df.describe())

# Check unique values in categorical columns (detect anomalies)
for col in ["service", "state", "attack_cat"]:
    print(f"Unique values in {col}: {df[col].unique()}")
