import pandas as pd

# Show all rows (disable truncation)
pd.set_option('display.max_rows', None)

# Load and analyze
df = pd.read_csv(r'd:\4th semester\SE\project\Dataset\02-14-2018_balanced.csv')
unique_counts = df.nunique().sort_values()
print(unique_counts)
