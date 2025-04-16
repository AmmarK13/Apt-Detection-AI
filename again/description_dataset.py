import pandas as pd


pd.set_option('display.max_rows', None, 'display.max_columns', None)
# Load your dataset
input_path = r"D:\4th semester\SE\project\Dataset\removed_neg_inf.csv" # Replace with your actual path
df = pd.read_csv(input_path)

# Display the first few rows
print("📄 Preview of the dataset:\n")
print(df.head())

# Display dataset info
print("\nℹ️ Dataset info:\n")
(df.info())
