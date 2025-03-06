import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
dataset_path = 'csic_database.csv'  # Update with the correct path
data = pd.read_csv(dataset_path)

# 1. Preview the dataset
print("Dataset Preview:")
print(data.head())

# 2. Check the shape of the dataset
print("\nDataset Shape:")
print(data.shape)

# 3. Display column names
print("\nColumn Names:")
print(data.columns)

# 1. Check for missing values
print("\nMissing Values in Each Column:")
print(data.isnull().sum())

# 2. Inspect data types
print("\nData Types of Columns:")
print(data.dtypes)

# Drop columns with more than 50% missing values
threshold = 0.5 * len(data)
data = data.dropna(axis=1, thresh=threshold)
print("\nRemaining Columns After Dropping:")
print(data.columns)

# Separate numerical and categorical columns
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = data.select_dtypes(include=['object']).columns

# 1. Plot distributions for numerical columns
for col in numerical_cols:
    plt.figure(figsize=(8, 5))
    sns.histplot(data[col], kde=True, bins=30, color='blue')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

# 2. Bar charts for categorical columns
for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    data[col].value_counts().plot(kind='bar', color='orange')
    plt.title(f"Category Distribution in {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()
