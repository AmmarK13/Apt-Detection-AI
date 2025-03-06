import polars as pl

# Load train and test data
train_file = "data/UNSW_NB15_training-set.parquet"
test_file = "data/UNSW_NB15_testing-set.parquet"

train_df = pl.read_parquet(train_file)
test_df = pl.read_parquet(test_file)

# Print schema (column names & types)
print("Train Schema:\n", train_df.schema)
print("\nTest Schema:\n", test_df.schema)

# Basic statistics
print("\nTrain Data Summary:\n", train_df.describe())
print("\nTest Data Summary:\n", test_df.describe())

# Check for column differences
train_columns = set(train_df.columns)
test_columns = set(test_df.columns)

if train_columns != test_columns:
    print("\n⚠️ Warning: Train and Test have different columns!")
    print("Train-only columns:", train_columns - test_columns)
    print("Test-only columns:", test_columns - train_columns)

# Check unique values per column
for col in train_df.columns:
    print(f"\n{col} - Unique values in Train:", train_df[col].n_unique())
    print(f"{col} - Unique values in Test:", test_df[col].n_unique())

# Save cleaned output for reference
print("writing csv")
train_df.head(5).write_csv("eda/train_sample.csv")
test_df.head(5).write_csv("eda/test_sample.csv")
print("csv done")

print("\n✅ Variable understanding completed. Sample data saved.")
