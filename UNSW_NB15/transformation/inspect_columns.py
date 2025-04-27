import pandas as pd

# def generate_feature_list(file_path: str):
#     df = pd.read_csv(file_path)
#     columns = df.columns.tolist()

#     print("[")
#     for col in columns:
#         print(f"    '{col}',")
#     print("]")

# if __name__ == "__main__":
#     # Example usage:
#     file_path = "UNSW_NB15/attempt_3/cleaned_attempt3/4_remove_features.csv"
#     generate_feature_list(file_path)
 

# def print_unique_values(file_path):
#     # Load the file
#     df = pd.read_csv(file_path)
    
#     # Print unique values for 'service' column
#     if 'service' in df.columns:
#         print("Unique values in 'service':")
#         print(df['service'].unique())
#     else:
#         print("'service' column not found.")
    
#     print("\n" + "-"*50 + "\n")
    
#     # Print unique values for 'state' column
#     if 'state' in df.columns:
#         print("Unique values in 'state':")
#         print(df['state'].unique())
#     else:
#         print("'state' column not found.")

# # Example usage:
# print_unique_values('UNSW_NB15/attempt_3/cleaned_attempt3/1_missingVals.csv')



import pandas as pd

def get_most_frequent_values(file_path: str):
    df = pd.read_csv(file_path)

    if 'service' in df.columns:
        most_freq_service = df['service'].mode()[0]
        print(f"Most frequent value in 'service': {most_freq_service}")
    
    if 'state' in df.columns:
        most_freq_state = df['state'].mode()[0]
        print(f"Most frequent value in 'state': {most_freq_state}")

# Example usage
get_most_frequent_values('UNSW_NB15/attempt_3/cleaned_attempt3/1_missingVals.csv')  # replace 'your_file.csv' with your real file path
