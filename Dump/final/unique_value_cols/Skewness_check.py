import pandas as pd

# Function to check skewness and remove skewed columns
def remove_skewed_columns(df, label_col='Label', skew_threshold=1.0):
    # Select columns that have only 2 unique values (except 'Label' column)
    binary_cols = [col for col in df.columns if df[col].nunique() == 2 and col != label_col]
    
    # Create a dictionary to store skewness values
    skewness_results = {}
    cols_to_remove = []  # List to store columns to be removed
    
    for col in binary_cols:
        skewness = df[col].skew()
        skewness_results[col] = skewness
        
        # If skewness is beyond the threshold, add column to removal list
        if abs(skewness) > skew_threshold:
            cols_to_remove.append(col)
    
    # Drop the columns with high skewness
    df.drop(columns=cols_to_remove, inplace=True)
    
    # Display the results
    print(f"Removed columns with skewness beyond {skew_threshold}:")
    for col in cols_to_remove:
        print(f"  {col} (Skewness: {skewness_results[col]})")
    
    return df, skewness_results

# Example usage
file_path = r"D:\4th semester\SE\project\Dataset\Dropped_1_unique.csv"  # Replace with your file path
df = pd.read_csv(file_path)

# Call the function to remove skewed columns
df, skewness_results = remove_skewed_columns(df)

# Optionally, save the cleaned dataset
output_path = r"D:\4th semester\SE\project\Dataset\Skew_free.csv"
df.to_csv(output_path, index=False)

print("Cleaned dataset saved at:", output_path)
