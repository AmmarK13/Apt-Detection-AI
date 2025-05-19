import pandas as pd

# Load the dataset
file_path = r"D:\4th semester\SE\project\Dataset\Cleaned_dataset_noneg.csv"
df = pd.read_csv(file_path)

# Columns with many negative values
high_neg_cols = ['Init Fwd Win Byts', 'Init Bwd Win Byts']

# Convert Label to numeric: Attack = 1, Benign = 0
df['Label_Numeric'] = df['Label'].map({'Benign': 0, 'Attack': 1})

# Ensure the relevant columns are numeric
df[high_neg_cols] = df[high_neg_cols].apply(pd.to_numeric, errors='coerce')

# Correlation with Label
correlation_with_label = df[high_neg_cols + ['Label_Numeric']].corr()['Label_Numeric'].drop('Label_Numeric')

# Display the results
print("\n📊 Correlation of high-negative columns with 'Label':\n")
print(correlation_with_label)
