import pandas as pd
 
df = pd.read_csv('UNSW_NB15/cleaned_attempt2/1_encoded_data.csv')

# Inspect the columns, their names, and the total number of columns
print("Column Names:", df.columns.tolist())  # List column names
print("Total Number of Columns:", df.shape[1])  # Number of columns
print("First 5 rows of the DataFrame:\n", df.head())  # Display first 5 rows for inspection

# Check for missing values in the DataFrame
print("\nMissing values per column:\n", df.isna().sum())

# Check the data types of each column
print("\nData Types of Columns:\n", df.dtypes)

# Summary of the DataFrame
print("\nData Summary:\n", df.describe())

# Check if any categorical columns were incorrectly encoded
print("\nUnique values in each column:")
for column in df.columns:
    print(f"{column}: {df[column].nunique()} unique values")




# step 4    ====== label imbalance
df = pd.read_csv('UNSW_NB15/cleaned_attempt2/2_normalised_data.csv')
# Check counts of 0s and 1s in the 'label' column
label_counts = df['label'].value_counts()
print("Label distribution:")
print(label_counts)



# # step 4 after ==== analyze ====
import pandas as pd

# Load the cleaned and balanced file
df_balanced = pd.read_csv("UNSW_NB15/cleaned_attempt2/3_remove_features.csv")

# Display shape of the dataset
print("Shape of dataset:", df_balanced.shape)

# Label distribution (counts)
label_counts = df_balanced['label'].value_counts()
print("\nLabel distribution (count):")
print(label_counts)

# Label distribution (percentages)
label_percent = df_balanced['label'].value_counts(normalize=True) * 100
print("\nLabel distribution (percentage):")
print(label_percent.round(2))

