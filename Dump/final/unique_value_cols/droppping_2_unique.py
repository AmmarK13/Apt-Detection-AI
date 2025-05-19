import pandas as pd

# Load your dataset
df = pd.read_csv(r"D:\4th semester\SE\project\Dataset\Dropped_1_unique.csv")  # Replace with the actual path to your dataset

# Drop columns that are less useful based on previous analysis
cols_to_drop = ['FIN Flag Cnt', 'Fwd PSH Flags', 'ECE Flag Cnt']
df.drop(columns=cols_to_drop, inplace=True)

# Show the updated DataFrame
print(f"Columns dropped: {cols_to_drop}")
print(df.head())

# Save the cleaned dataset to a new file
output_path = r"D:\4th semester\SE\project\Dataset\Cleaned_Dataset.csv"
df.to_csv(output_path, index=False)  # Replace with your desired output path
