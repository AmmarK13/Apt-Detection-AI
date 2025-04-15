import pandas as pd

def clean_low_variance_columns(input_path, output_path):
    # Load dataset
    df = pd.read_csv(input_path)
    
    # Identify columns with only one unique value
    unique_counts = df.nunique()
    low_variance_cols = unique_counts[unique_counts == 1].index.tolist()
    
    # Drop those columns
    df_cleaned = df.drop(columns=low_variance_cols)
    
    # Save cleaned dataset
    df_cleaned.to_csv(output_path, index=False)
    
    # Output some info
    print(f"Removed {len(low_variance_cols)} column(s) with only one unique value.")
    print(f"Cleaned dataset saved to: {output_path}")

# Example usage
input_file =r'd:\4th semester\SE\project\Dataset\02-14-2018_balanced.csv'
output_file = r'd:\4th semester\SE\project\Dataset\02-14-2018_balanced_removed_cols.csv'
clean_low_variance_columns(input_file, output_file)
