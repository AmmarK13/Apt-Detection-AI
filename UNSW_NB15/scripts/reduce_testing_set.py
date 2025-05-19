import polars as pl

# Load dataset from CSV
df = pl.read_csv("UNSW_NB15/data/UNSW_NB15_testing-set.csv")

# Randomly select 50,000 rows
sampled_df = df.sample(n=80000, shuffle=True, seed=42)  # Shuffle ensures randomness

# Save as CSV
csv_path = "UNSW_NB15/cleaned_data/reduced_testing_set.csv"
sampled_df.write_csv(csv_path)

print(f"Saved 80,000 random rows to {csv_path}")


