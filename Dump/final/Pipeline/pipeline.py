import pandas as pd
import os

def process_dataset(input_path, output_path):
    # Load dataset
    df = pd.read_csv(input_path)

    # --- Step 1: Convert 'Timestamp' to datetime ---
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S')
        df['Time_Seconds'] = df['Timestamp'].dt.hour * 3600 + df['Timestamp'].dt.minute * 60 + df['Timestamp'].dt.second
        df.drop(columns=['Timestamp'], inplace=True)

    # --- Step 2: Create 'Time_Diff' column ---
    df = df.sort_values(by='Time_Seconds')  # sort by time just in case
    df['Time_Diff'] = df['Time_Seconds'].diff().fillna(0)

    # --- Step 3: Drop columns with only 1 unique value ---
    cols_with_one_val = [col for col in df.columns if df[col].nunique() == 1]
    df.drop(columns=cols_with_one_val, inplace=True)
    
    print(f"Dropped {len(cols_with_one_val)} columns with only one unique value:")
    for col in cols_with_one_val:
        print(f" - {col}")

    # --- Step 4: Save to output path ---
    df.to_csv(output_path, index=False)
    print(f"\n✅ Cleaned dataset saved to: {output_path}")

# Example usage
if __name__ == "__main__":
    input_path = input("Enter full path of your input CSV file: ").strip()
    output_path = input("Enter full path for your output cleaned file: ").strip()

    if os.path.exists(input_path):
        process_dataset(input_path, output_path)
    else:
        print("❌ Input file does not exist. Please check the path.")
