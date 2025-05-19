import pandas as pd

# === USER INPUT PATHS ===
input_path = r"D:\4th semester\SE\project\Dataset\New_Dataset_52_48.csv"  # <-- change this to your file path
output_path = r"D:\4th semester\SE\project\Dataset\Processed_Timestamp_Encoded.csv"  # <-- where to save result

# === LOAD DATA ===
df = pd.read_csv(input_path)


df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S')


df['Hour'] = df['Timestamp'].dt.hour
df['Minute'] = df['Timestamp'].dt.minute
df['Second'] = df['Timestamp'].dt.second
df['Time_Seconds'] = df['Hour'] * 3600 + df['Minute'] * 60 + df['Second']
df['Time_Diff'] = df['Timestamp'].diff().dt.total_seconds().fillna(0)


df.drop(columns=['Timestamp'], inplace=True)

df.to_csv(output_path, index=False)
print(f"✅ File saved at: {output_path}")
