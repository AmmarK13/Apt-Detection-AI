import pandas as pd

# Load the data
df = pd.read_csv("UNSW_NB15/cleaned_data/reduced_training_set.csv")

# Count total rows and missing 'service' entries
total_rows = df.shape[0]
missing_service_count = df[df['service'] == '-'].shape[0]
print(f"Total number of rows: {total_rows}")
print(f"Number of rows with '-' in 'service' column: {missing_service_count}")

# Drop rows where 'service' is '-'
df_cleaned = df[df['service'] != '-']
print(f"Rows remaining after dropping: {df_cleaned.shape[0]}")

# Save to a new CSV
output_path = "UNSW_NB15/cleaned_attempt3/1_missingVals.csv"
df_cleaned.to_csv(output_path, index=False)
print(f"Cleaned dataset saved to: {output_path}")
