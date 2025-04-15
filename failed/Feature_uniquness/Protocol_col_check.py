import pandas as pd

def check_protocol_attack_benign_balance(file_path):
    df = pd.read_csv(file_path)
    
    # Ensure 'Protocol' and 'Label' columns are in the dataframe
    if 'Protocol' in df.columns and 'Label' in df.columns:
        # Group by Protocol and Label to calculate percentages
        protocol_label_counts = df.groupby(['Protocol', 'Label']).size().unstack(fill_value=0)
        
        # Calculate the percentage of Benign and Attack for each Protocol value
        protocol_label_percentage = protocol_label_counts.div(protocol_label_counts.sum(axis=1), axis=0) * 100
        
        print("🧠 Protocol to Attack/Benign Distribution (in percentage):\n")
        print(protocol_label_percentage)
        
        # Remove rows where 'Protocol' is either 0 or 17
        df = df[~df['Protocol'].isin([0, 17])]

        # Save the modified DataFrame back to a new CSV file
        output_file_path = file_path.replace('.csv', '_modified.csv')  # Generate new file path
        df.to_csv(output_file_path, index=False)
        print(f"✅ The modified file has been saved as: {output_file_path}")
        
    else:
        print("⚠️ 'Protocol' or 'Label' column not found.")

# Example usage
file_path = r"d:\4th semester\SE\project\Dataset\02-14-2018_balanced_removed_cols_free_2skew.csv"
check_protocol_attack_benign_balance(file_path)
