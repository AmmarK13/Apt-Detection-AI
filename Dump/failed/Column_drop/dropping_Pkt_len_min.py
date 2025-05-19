import pandas as pd

file_path=r"d:\4th semester\SE\project\Dataset\cleaned_up_modified_no_protocol.csv"

df = pd.read_csv(file_path)


# Drop the 'Protocol' column from the DataFrame
df = df.drop(columns=['Pkt Len Min'])

# Save the modified DataFrame back to a new CSV file
output_file_path = file_path.replace('.csv', '_modified_no_protocol.csv')  # Generate new file path
df.to_csv(output_file_path, index=False)
print(f"✅ The modified file without 'Protocol' column has been saved as: {output_file_path}")
