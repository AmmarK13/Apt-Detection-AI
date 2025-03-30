import polars as pl  

# Load the dataset
file_path = "UNSW_NB15/cleaned_data/missingVals-testing.csv"
df = pl.read_csv(file_path)

# Count and remove exact duplicates
num_duplicates = df.is_duplicated().sum()  
print(f"Duplicate rows: {num_duplicates}")

df = df.unique()
print(f"Dataset shape after removing duplicates: {df.shape}")

# Print class distribution
print("Class distribution after removing duplicates:")
df_pandas = df.to_pandas()
print(df_pandas["label"].value_counts())

# --- NEAR DUPLICATES PROCESSING ---
print("\nNEAR DUPLICATES")
columns_to_exclude = [
    "dur", "rate", "sload", "dload", "sinpkt", "dinpkt", "sjit", "djit",
    "tcprtt", "synack", "ackdat", "trans_depth"
]

# Drop only the columns that exist
existing_columns = [col for col in columns_to_exclude if col in df.columns]
filtered_df = df.drop(existing_columns)

# Identify near-duplicates
near_dup_mask = filtered_df.is_duplicated()
num_near_duplicates = near_dup_mask.sum()
print(f"Near-duplicate rows (excluding variable network timing features): {num_near_duplicates}")

# Get indices of rows to **keep**
rows_to_keep = ~near_dup_mask

# Remove near-duplicates
filtered_df = filtered_df.filter(rows_to_keep)
print(f"Dataset shape after removing near-duplicates: {filtered_df.shape}")

# Filter excluded columns **to match filtered_df**
excluded_data = df[existing_columns].filter(rows_to_keep) if existing_columns else None

# Merge back excluded columns
if excluded_data is not None and excluded_data.shape[0] == filtered_df.shape[0]:
    final_df = filtered_df.with_columns(excluded_data)
    print("\nmerged back successfully")
    
else:
    print("Error: Could not merge back excluded columns due to row mismatch.")
    final_df = filtered_df  # Fallback to using filtered data only


print(f"Dataset shape now: {final_df.shape}")


# --- VERIFICATION ---
print("\n\nVERIFICATION")
print(f"Remaining exact duplicates: {final_df.is_duplicated().sum()}")
print(final_df["label"].value_counts())
print(f"Final dataset shape: {final_df.shape}")

# Save cleaned dataset with all columns
final_df.write_csv("UNSW_NB15/cleaned_data/testing_dups_removed.csv")
print("cleaned csv saved")