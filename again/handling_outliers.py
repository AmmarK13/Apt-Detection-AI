import pandas as pd
import numpy as np
from scipy import stats
pd.set_option('display.max_rows', None, 'display.max_columns', None)
# === File Path ===
file_path = r"D:\4th semester\SE\project\Dataset\cleaned_up_dataset.csv"  # 🔁 Replace with your actual file path

# === Load the dataset ===
df = pd.read_csv(file_path)

# === Function to filter outliers using Z-score ===
def filter_outliers_zscore(data, threshold):
    z_scores = np.abs(stats.zscore(data))
    outlier_mask = (z_scores > threshold).any(axis=1)
    return data[~outlier_mask], data[outlier_mask]

# === Threshold ===
threshold = 7

# === Loop through the columns and filter outliers ===
filtered_cols = []
removed_outliers = []

for col in df.columns:
    if col != 'Label':
        filtered_col, outliers = filter_outliers_zscore(df[[col]], threshold)
        filtered_cols.append(filtered_col)
        removed_outliers.append(outliers)

# === Combine filtered and outlier dataframes ===
df_filtered = pd.concat(filtered_cols, axis=1)
df_outliers = pd.concat(removed_outliers, axis=1)

# === Output Summary ===
print(f'\nOriginal Data Shape: {df.shape}')
print('Outlier removal summary:')
print(f'{df_outliers.shape[0]} outlier rows would be removed\n')

# === Print previews ===
print('\nOriginal dataframe:')
print(df.head())

# Assign filtered numeric columns back
columns = [col for col in df.columns if col != 'Label']
df.loc[:, columns] = df_filtered.loc[:, columns]

# === Save the filtered DataFrame to a new file ===
output_path = r"D:\4th semester\SE\project\Dataset\filtered_dataset.csv"
df.to_csv(output_path, index=False)

print(f"\n💾 Filtered dataset saved to: {output_path}")

# === Print final filtered dataframe ===
print('\nFiltered dataframe:')
print(df.head())

# === Print removed outliers ===
print('\nRemoved outliers:')
print(df_outliers.head())

# === Show which attack types had rows removed (if any) ===
values_orig = df.loc[df.index.isin(df_outliers.index), 'Label']
print(f'\nAttack types removed in outliers:\n{values_orig.value_counts()}')
