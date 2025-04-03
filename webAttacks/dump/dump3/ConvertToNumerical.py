import pandas as pd

# Load the cleaned and preprocessed dataset
df = pd.read_csv("data/Cleaned_Preprocessed.csv")

# Define the target column
target_col = 'classification'

# Convert all categorical features (all columns except the target) using one-hot encoding
df_encoded = pd.get_dummies(df.drop(target_col, axis=1), drop_first=True)
# Append the target column back
df_encoded[target_col] = df[target_col]

# Save the encoded data to a new CSV file
output_file = "data/Cleaned_Preprocessed_Numerical.csv"
df_encoded.to_csv(output_file, index=False)
print(f"Encoded data saved to '{output_file}'")
