import pandas as pd

def print_min_max(file_path):
    df = pd.read_csv(file_path)

    print("📊 Min and Max values for each column:\n")
    min_max_df = pd.DataFrame({
        'Min': df.min(numeric_only=True),
        'Max': df.max(numeric_only=True)
    })
    print(min_max_df)

# Example usage
file_path = r"d:\4th semester\SE\project\Dataset\balanced_dataset_v2_v3.csv"
print_min_max(file_path)
