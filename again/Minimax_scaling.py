import pandas as pd
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_rows', None, 'display.max_columns', None)

# === File Path ===
file_path = r"D:\4th semester\SE\project\Dataset\filtered_dataset_no_nan.csv"  # 🔁 Replace with your actual file path

# === Load the dataset ===
df = pd.read_csv(file_path)

# === Define the columns to scale ===
columns = [col for col in df.columns if col != 'Label']

# === Apply MinMaxScaler ===
min_max_scaler = MinMaxScaler().fit(df[columns])
df[columns] = min_max_scaler.transform(df[columns])

# === Output Path ===
output_path = r"D:\4th semester\SE\project\Dataset\scaled_cleaned_up_dataset.csv"  # 🔁 Specify your desired output path

# === Save the scaled dataset ===
df.to_csv(output_path, index=False)

# === Print the first few rows of the scaled dataset ===
print(f"Scaled dataset saved at: {output_path}")
print(f"Scaled dataset shape: {df.shape}")
print(df.head())





