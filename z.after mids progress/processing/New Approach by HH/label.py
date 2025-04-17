import pandas as pd

df = pd.read_csv("dataset_balanced.csv")
print(df['Label'].value_counts())
