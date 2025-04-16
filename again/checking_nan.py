import pandas as pd

pd.set_option('display.max_rows', None, 'display.max_columns', None)

# Load your dataset (replace with the actual file path)
file_path = r"D:\4th semester\SE\project\Dataset\filtered_dataset_no_nan.csv"
df = pd.read_csv(file_path)

# Step 1: Convert object columns (except 'Label') to numeric, replace errors with NaN
for col in df.columns:
    if df[col].dtype == 'object' and col != 'Label':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Step 2: Count NaN values in each column
count_NA = df.isna().sum()

# Print the results
print(f"Missing Values Count (NaN) per Column:\n{count_NA}")
