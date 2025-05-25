import pandas as pd

file_path=r"D:\4th semester\SE\project\Dataset\balanced_dataset_v2.csv"

df = pd.read_csv(file_path)


# Drop the 'Protocol' column from the DataFrame
df = df.drop(columns=['Down/Up Ratio','Fwd Pkt Len Min','Bwd Pkt Len Min'])

# Save the modified DataFrame back to a new CSV file
output_file_path = file_path.replace('.csv', '_v3.csv')  # Generate new file path
df.to_csv(output_file_path, index=False)
print(f"✅ The modified file without 'Protocol' column has been saved as: {output_file_path}")
