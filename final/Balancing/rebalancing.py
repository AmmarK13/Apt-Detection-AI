import pandas as pd

def rebalance_labels(file_path, output_path, target_attack_ratio=0.48):
    df = pd.read_csv(file_path)

    # Split data by label
    attack_df = df[df['Label'] == 'Attack']
    benign_df = df[df['Label'] == 'Benign']

    total_desired = len(df)
    target_attack_count = int(total_desired * target_attack_ratio)
    target_benign_count = total_desired - target_attack_count

    # Adjust only if imbalance exists
    if len(attack_df) > target_attack_count:
        attack_df = attack_df.sample(target_attack_count, random_state=42)
    if len(benign_df) > target_benign_count:
        benign_df = benign_df.sample(target_benign_count, random_state=42)

    # Combine and shuffle
    balanced_df = pd.concat([attack_df, benign_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Save the result
    balanced_df.to_csv(output_path, index=False)
    print(f"✅ Dataset rebalanced and saved to: {output_path}")

    # Print new distribution
    label_counts = balanced_df['Label'].value_counts(normalize=True) * 100
    print("\n📊 New Label Distribution:")
    print(label_counts.round(2))

# Example usage
file_path = r"d:\4th semester\SE\project\Dataset\cleaned_up_version2.csv"
output_path = r"d:\4th semester\SE\project\Dataset\balanced_dataset_v2.csv"
rebalance_labels(file_path, output_path)
