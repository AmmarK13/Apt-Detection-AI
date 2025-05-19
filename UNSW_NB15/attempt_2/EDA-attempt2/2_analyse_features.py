import pandas as pd

# Load your encoded dataset
file_path = "UNSW_NB15/cleaned_attempt2/1_encoded_data.csv"  
df = pd.read_csv(file_path)

# Show basic info
print("Total columns:", df.shape[1])
print("\nData types:")
print(df.dtypes)

print("\nNumber of unique values per column:")
print(df.nunique().sort_values())

# Optional: Show columns with more than 20 unique values (likely continuous)
print("\nLikely continuous features (more than 20 unique values):")
likely_continuous = df.nunique()[df.nunique() > 20]
print(likely_continuous)
