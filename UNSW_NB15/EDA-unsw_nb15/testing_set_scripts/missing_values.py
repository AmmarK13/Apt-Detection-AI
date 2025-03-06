import polars as pl

# Load the dataset
file_path = "UNSW_NB15/cleaned_data/reduced_testing_set.csv"
df = pl.read_csv(file_path)

# Count missing values per column
missing_counts = df.null_count()

# Ensure all columns are visible
pl.Config.set_tbl_cols(len(df.columns))

# Print missing value summary
print("Missing Values in Each Column:")
print(missing_counts)