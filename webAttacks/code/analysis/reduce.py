import pandas as pd
import numpy as np
import os

def balance_dataset(input_path, output_path, target_col='classification', target_ratios={'0': 0.52, '1': 0.48}):
    # Read the input dataset
    df = pd.read_csv(input_path)
    
    # Get current class counts
    class_counts = df[target_col].value_counts()
    
    # Calculate target counts based on desired ratios
    total_target_size = int(min(class_counts) / min(target_ratios.values()))
    target_counts = {k: int(v * total_target_size) for k, v in target_ratios.items()}
    
    # Sample from each class to achieve target distribution
    balanced_dfs = []
    for class_label, target_count in target_counts.items():
        class_data = df[df[target_col] == int(class_label)]
        if len(class_data) > target_count:
            balanced_dfs.append(class_data.sample(n=target_count, random_state=42))
        else:
            balanced_dfs.append(class_data)
    
    # Combine balanced datasets
    balanced_df = pd.concat(balanced_dfs)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Save balanced dataset to new CSV file
        balanced_df.to_csv(output_path, index=False)
        print(f"Successfully saved balanced dataset to: {output_path}")
    except Exception as e:
        print(f"Error saving file: {str(e)}")
        raise
    
    return balanced_df

if __name__ == "__main__":
    # Define input and output paths
    input_path = r"D:\University\Software Engineering\Project\data\csic_cleaned.csv"
    output_path = r"D:\University\Software Engineering\Project\data\csic_reduced.csv"
    
    # Define target ratios for balancing
    target_ratios = {'0': 0.52, '1': 0.48}
    
    # Balance the dataset
    balanced_df = balance_dataset(
        input_path=input_path,
        output_path=output_path,
        target_ratios=target_ratios
    )
    
    # Print summary statistics
    print("Dataset balancing completed:")
    print(f"Total samples: {len(balanced_df)}")
    print("Class distribution:")
    print(balanced_df['classification'].value_counts(normalize=True))
