import pandas as pd
import numpy as np

# Load your dataset
file_path = r"D:\4th semester\SE\project\Dataset\filtered_dataset_no_nan.csv"
df = pd.read_csv(file_path)

# Select only numeric columns to avoid TypeError
numeric_df = df.select_dtypes(include=[np.number])

# Count +inf and -inf values
count_inf = np.isposinf(numeric_df).sum().sum()
count_neg_inf = np.isneginf(numeric_df).sum().sum()
print(f"+Inf count: {count_inf}")
print(f"-Inf count: {count_neg_inf}")

# Replace +inf and -inf with NaN across entire DataFrame
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with any NaN values
df.dropna(inplace=True)

# Save the cleaned dataset
output_path = r"D:\4th semester\SE\project\Dataset\removed_neg_inf.csv"
df.to_csv(output_path, index=False)
print(f"✅ Cleaned dataset saved at: {output_path}")
