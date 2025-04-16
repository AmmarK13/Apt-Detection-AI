import pandas as pd

# Load your dataset
file_path = r"D:\4th semester\SE\project\Dataset\removed_neg_inf.csv"# Replace with your actual file path
df = pd.read_csv(file_path)

# Count the values in the 'Label' column
count = df['Label'].value_counts()

# Print the result
print(f"Value counts for 'Label' column:\n{count}")
