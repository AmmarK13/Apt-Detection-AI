import pandas as pd
import numpy as np


pd.set_option('display.max_rows', None, 'display.max_columns', None)
# === File Path ===
file_path = r"D:\4th semester\SE\project\Dataset\scaled_cleaned_up_dataset.csv" # 🔁 Replace with your actual file path

# === Load the dataset ===
df = pd.read_csv(file_path)

# === Define columns to exclude 'Label' ===
columns = [col for col in df.columns if col != 'Label']

# === Calculate correlation matrix ===
corr_matrix = df[columns].corr().abs()

# === Define threshold for high correlation ===
threshold = 0.90

# === Find features with high correlation ===
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

# === Output which features will be dropped ===
print(f"The following {len(to_drop)} features will be dropped due to high correlation: {to_drop}")

# === Drop the highly correlated features ===
df = df.drop(to_drop, axis=1)

# === Output Path ===
output_path = r"D:\4th semester\SE\project\Dataset\afewstepsleft.csv"  # 🔁 Specify your desired output path

# === Save the modified dataset ===
df.to_csv(output_path, index=False)

# === Print the first few rows of the modified dataset ===
print(f"Modified dataset saved at: {output_path}")
print(f"Modified dataset shape: {df.shape}")
print(df.head())
