import pandas as pd

def transform_to_match_reference(test_data_path, reference_data_path, output_transformed_path, fill_method='mean'):
    """
    Transforms test data to match the columns of the reference dataset.
    If a column is missing in the test data, it will be filled with the mean or sum (based on fill_method).
    
    :param test_data_path: Path to the test dataset.
    :param reference_data_path: Path to the reference (training) dataset.
    :param output_transformed_path: Path to save the transformed test dataset.
    :param fill_method: Method to fill missing columns, options: 'mean' or 'sum'. Default is 'mean'.
    """
    # Load the datasets
    reference_data = pd.read_csv(reference_data_path)
    test_data = pd.read_csv(test_data_path)

    # Step 1: Ensure all columns in reference data exist in test data
    reference_cols = reference_data.columns
    missing_cols = set(reference_cols) - set(test_data.columns)

    # Fill missing columns with the specified fill method
    for col in missing_cols:
        if fill_method == 'mean':
            # Fill with the mean of the reference column
            fill_value = reference_data[col].mean()
        elif fill_method == 'sum':
            # Fill with the sum of the reference column
            fill_value = reference_data[col].sum()
        else:
            raise ValueError("Invalid fill_method. Use 'mean' or 'sum'.")
        
        # Add the missing column to the test data
        test_data[col] = fill_value

    # Step 2: Reorder test data columns to match reference data
    test_data = test_data[reference_cols]

    # Step 3: Save the transformed test data
    test_data.to_csv(output_transformed_path, index=False)

    print(f"✅ Transformed data saved to {output_transformed_path}")

# Example usage
transform_to_match_reference(
    test_data_path=r"D:\4th semester\SE\project\Datasets\benign_ftp_bruteforce.csv",  # New data you want to process
    reference_data_path=r"C:\Users\user\.cache\kagglehub\datasets\ogguy11\apt-detection\versions\1\02-14-2018.csv",  # Reference training data used to train the model
    output_transformed_path=r'D:\4th semester\SE\project\Datasets\transformed_ftp_bruteforce.csv',  # Output file for transformed data
    fill_method='mean'  # Use 'mean' or 'sum' for filling missing columns
)
