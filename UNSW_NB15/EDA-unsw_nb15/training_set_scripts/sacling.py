import pandas as pd
import matplotlib.pyplot as plt


# Load the dataset (update the file path)
df = pd.read_csv("UNSW_NB15/cleaned_data/training_selected_features.csv")



# is scaling even needed?
# Display basic stats
print(df.describe())


# FEATURE RANGES
print("FEATURE RANGES")
for col in df.columns:
    print(f"{col}: Min={df[col].min()}, Max={df[col].max()}")


# FEATURE DISTRIBUTION
print("FEATURE DISTRIBUTION")
df.hist(figsize=(12, 8), bins=50)
plt.show()
