import pandas as pd

def inspect_limited_unique_columns(file_path, max_unique=7):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Get columns with more than 1 and up to max_unique unique values
    selected_cols = df.nunique()
    limited_unique_cols = selected_cols[(selected_cols > 1) & (selected_cols <= max_unique)].index

    # Print the values each of these columns can take
    print(f"\nColumns with 2 to {max_unique} unique values:\n")
    for col in limited_unique_cols:
        print(f"🔹 {col} ({df[col].nunique()} unique values): {sorted(df[col].dropna().unique())}")
    print("\nDone.")

# Example usage
file_path = r"d:\4th semester\SE\project\Dataset\02-14-2018_balanced_removed_cols.csv"
inspect_limited_unique_columns(file_path)
