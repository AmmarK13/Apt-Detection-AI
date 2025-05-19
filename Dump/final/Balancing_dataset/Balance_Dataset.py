import pandas as pd

def balance_by_time(input_csv_path, output_csv_path, benign_ratio=0.52, attack_ratio=0.48):
    """
    Balance the dataset by label (Benign/Attack) without shuffling to retain time patterns.
    """
    # Step 1: Load dataset
    df = pd.read_csv(input_csv_path)

    # Step 2: Label simplification
    df['Label'] = df['Label'].replace({
        'FTP-BruteForce': 'Attack',
        'SSH-Bruteforce': 'Attack'
    })

    # Step 3: Convert and sort by Timestamp
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp'])  # Drop invalid timestamps if any
    df = df.sort_values('Timestamp').reset_index(drop=True)

    # Step 4: Split by class
    benign_df = df[df['Label'] == 'Benign']
    attack_df = df[df['Label'] == 'Attack']

    # Step 5: Balance sizes
    total_desired = min(len(benign_df) / benign_ratio, len(attack_df) / attack_ratio)
    target_benign = int(total_desired * benign_ratio)
    target_attack = int(total_desired * attack_ratio)

    # Step 6: Take first N rows to preserve burst order
    balanced_df = pd.concat([
        benign_df.head(target_benign),
        attack_df.head(target_attack)
    ])

    # Step 7: Sort by time again
    balanced_df = balanced_df.sort_values('Timestamp').reset_index(drop=True)

    # Step 8: Save
    balanced_df.to_csv(output_csv_path, index=False)

    # Show distribution
    print("Balanced Label Distribution (in %):")
    print(balanced_df['Label'].value_counts(normalize=True) * 100)

# 🟢 Example usage
input_path = r"C:\Users\user\.cache\kagglehub\datasets\ogguy11\apt-detection\versions\1\02-14-2018.csv"
output_path = r"D:\4th semester\SE\project\Dataset\New_Dataset_52_48.csv"
balance_by_time(input_path, output_path)

