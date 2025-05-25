import pandas as pd

# Load your dataset
df = pd.read_csv(r"D:\4th semester\SE\project\Dataset\Cleaned_Dataset.csv")  # Replace with your actual file path

# Convert all columns to numeric, forcing errors to NaN (not a number)
df = df.apply(pd.to_numeric, errors='coerce')

# Find columns with negative values
negative_values_count = {}

# Iterate over each column to check for negative values
for column in df.columns:
    # Count the number of negative values in the column
    negative_count = (df[column] < 0).sum()
    if negative_count > 0:
        negative_values_count[column] = negative_count

# Display the results
if negative_values_count:
    print("Columns with negative values and their respective counts:")
    for col, count in negative_values_count.items():
        print(f"{col}: {count} negative values")
else:
    print("No negative values found in the dataset.")
