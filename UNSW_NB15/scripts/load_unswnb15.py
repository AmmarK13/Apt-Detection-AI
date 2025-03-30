import polars as pl

# Define file paths
train_file = "data/UNSW_NB15_training-set.parquet"
test_file = "data/UNSW_NB15_testing-set.parquet"

# Load data using Polars
train_df = pl.read_parquet(train_file)
test_df = pl.read_parquet(test_file)

# Display basic info
print(train_df.shape)   # Number of rows and columns
print(train_df.head(5))  # First 5 rows
print(train_df.schema)  # Column names and data types

print("done!")
