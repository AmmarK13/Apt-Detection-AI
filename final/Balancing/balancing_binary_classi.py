import pandas as pd

# Load dataset
file_path = 'your_file.csv'
df = pd.read_csv(file_path)

# Undo label encoding
df['Label'] = df['Label'].replace({0: 'Benign', 1: 'Attack', 2: 'Attack'})

# Shuffle the dataset for randomness
df = df.sample(frac=1, random_state=42)

# Desired total
total_rows = 10000
benign_target = int(total_rows * 0.52)  # 52%
attack_target = total_rows - benign_target  # 48%

# Split and sample
benign_df = df[df['Label'] == 'Benign'].sample(n=benign_target, random_state=42)
attack_df = df[df['Label'] == 'Attack'].sample(n=attack_target, random_state=42)

# Combine and shuffle again
balanced_df = pd.concat([benign_df, attack_df]).sample(frac=1, random_state=42).reset_index(drop=True)

# Output new distribution
print(balanced_df['Label'].value_counts(normalize=True) * 100)
print(f"\nTotal rows: {len(balanced_df)}")

# Optionally save
balanced_df.to_csv('balanced_dataset.csv', index=False)
