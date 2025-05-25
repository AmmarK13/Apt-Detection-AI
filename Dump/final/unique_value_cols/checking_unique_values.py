import pandas as pd

def count_unique_values(file_path):
    # Display full column list without truncation
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_rows', None)

    # Load the dataset
    df = pd.read_csv(file_path)

    # Count unique values for each column
    unique_counts = df.nunique()

    # Print full result
    print("🔍 Unique value counts per column:\n")
    print(unique_counts.sort_values(ascending=False))

    return unique_counts

# Example usage:
file_path = r"D:\4th semester\SE\project\Dataset\Processed_Timestamp_Encoded.csv"
count_unique_values(file_path)
