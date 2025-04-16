import pandas as pd

# === File Path ===
file_path = r"D:\4th semester\SE\project\Dataset\filtered_dataset.csv"  # 🔁 Replace with your actual file path

# === Load the dataset ===
df = pd.read_csv(file_path)

# === Drop NaN values from the dataset ===
df.dropna(inplace=True)

# === Output Path ===
output_path = r"D:\4th semester\SE\project\Dataset\filtered_dataset_no_nan.csv"  # 🔁 Specify your desired output path

# === Save the cleaned dataset ===
df.to_csv(output_path, index=False)

# === Print the shape of the cleaned dataset ===
print(f"Cleaned dataset shape: {df.shape}")
print(f"Cleaned dataset saved at: {output_path}")
