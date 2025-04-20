import pandas as pd

def dataset_summary(file_path):
    """
    Summarizes the structure and details of the dataset.
    
    Parameters:
        file_path (str): Path to the dataset file (CSV).
    """
    try:
        # Load dataset
        data = pd.read_csv(file_path)
        
        # Dataset shape
        print("Dataset Shape:")
        print(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
        print("\n")
        
        # Column data types and missing values
        print("Column Data Types and Missing Values:")
        print(data.dtypes)
        print("\nMissing Values Count:")
        print(data.isnull().sum())
        print("\n")
        
        # Summary statistics
        print("Summary Statistics:")
        print(data.describe(include='all').transpose())
        print("\n")
        
        # Data preview
        print("First 5 Rows of the Dataset:")
        print(data.head())
        
    except Exception as e:
        print(f"An error occurred: {e}")

# Usage Example
# Replace 'your_dataset.csv' with the path to your dataset
file_path = 'D:/University/Software Engineering/Project/data/csic_reduced_minmaxScaled.csv'
dataset_summary(file_path)
