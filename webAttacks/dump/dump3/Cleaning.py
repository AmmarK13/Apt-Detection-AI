import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("data/csic_database.csv")

# Display only the column headers
print("Columns:", df.columns)

# Get info about columns, data types, and missing values
info_table = pd.DataFrame({
    "Column": df.columns,
    "Data Type": [df[col].dtype for col in df.columns],
    "Missing Values": [df[col].isnull().sum() for col in df.columns]
})
print(info_table)

# Summary statistics for numerical columns
summary_table = df.describe().transpose()
print(summary_table)

# Drop the unlabeled header column if it exists (commonly "Unnamed: 0")
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)
    print("\nDropped 'Unnamed: 0' column.")

# Drop columns with more than 70% missing values
missing_percentage = df.isnull().mean()  # fraction of missing values per column
columns_to_drop = missing_percentage[missing_percentage > 0.70].index
df = df.drop(columns_to_drop, axis=1)
print("\nDropped columns with more than 70% missing values:", columns_to_drop.tolist())

# Check for missing values after dropping columns
missing_values = df.isnull().sum()
print("\nMissing values after dropping columns:\n", missing_values)

# Fill missing values for numerical columns with the mean
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col] = df[col].fillna(df[col].mean())


# Fill missing values for categorical columns with the mode
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Remove duplicate rows (if any)
df = df.drop_duplicates()
print("\nRemoved duplicate rows.")

# If you have categorical columns like 'attack_type', encode them.
if 'attack_type' in df.columns:
    le = LabelEncoder()
    df['attack_type'] = le.fit_transform(df['attack_type'])

# Optionally, print a final summary of the cleaned data
print("Cleaned DataFrame info:")
print(df.info())

# Save the cleaned dataset to a new CSV file
df.to_csv("data/Cleaned_Preprocessed.csv", index=False)
print("Cleaned data saved to 'data/Cleaned_Preprocessed.csv'")