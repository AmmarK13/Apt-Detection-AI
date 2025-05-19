import pandas as pd

# Load the dataset
file_path = r"D:\4th semester\SE\project\Dataset\Cleaned_Dataset.csv"
df = pd.read_csv(file_path)

# List of columns with a few negative values
cols_with_few_negatives = [
    'Flow Duration',
    'Flow Pkts/s',
    'Flow IAT Mean',
    'Flow IAT Max',
    'Flow IAT Min',
    'Fwd IAT Tot',
    'Fwd IAT Mean',
    'Fwd IAT Max',
    'Fwd IAT Min',
    'Time_Diff'
]

# Replace negative values with 0 in those columns
for col in cols_with_few_negatives:
    if col in df.columns:
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            print(f"Replacing {neg_count} negative values in '{col}' with 0.")
            df[col] = df[col].apply(lambda x: 0 if x < 0 else x)

# Save the cleaned version (optional)
save_path=r"D:\4th semester\SE\project\Dataset\Cleaned_dataset_noneg.csv"
if save_path:
    df.to_csv(save_path, index=False)
    print(f"\n✅ Cleaned dataset saved to: {save_path}")
else:
    print("\n❗ Changes made in memory only. Not saved to disk.")
