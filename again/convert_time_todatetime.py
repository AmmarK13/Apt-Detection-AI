import pandas as pd

# Load your dataset
file_path = r"D:\4th semester\SE\project\Dataset\removed_neg_inf.csv" # Replace with your actual file path
df = pd.read_csv(file_path)

# Convert 'Timestamp' to DateTime format
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M:%S', errors='coerce')

# Now convert 'Timestamp' to seconds since Unix epoch
df['Timestamp'] = (df['Timestamp'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

# Save the cleaned dataset to a new file
output_path = r"D:\4th semester\SE\project\Dataset\cleaned_up_dataset.csv"  # Replace with your desired output file path
df.to_csv(output_path, index=False)

# Optional: Print to confirm
print(f"✅ Cleaned dataset saved at: {output_path}")
