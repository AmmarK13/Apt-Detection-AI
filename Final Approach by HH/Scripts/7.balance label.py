import pandas as pd
from sklearn.utils import resample

# Load dataset
df = pd.read_csv("dataset_scaled_cleaned.csv")

# Count label distribution
label_counts = df['Label'].value_counts()
total = label_counts.sum()
label_percentages = label_counts / total * 100

print("🔍 Label Distribution Before Balancing:")
for label, percent in label_percentages.items():
    print(f"  ➤ Label {int(label)}: {label_counts[label]} rows ({percent:.2f}%)")

# Check if it's imbalanced
min_percent = label_percentages.min()
max_percent = label_percentages.max()

if min_percent < 45 or max_percent > 55:
    print("\n⚖️ Label is imbalanced. Balancing now...")

    # Separate majority and minority classes
    majority_label = label_counts.idxmax()
    minority_label = label_counts.idxmin()

    df_majority = df[df['Label'] == majority_label]
    df_minority = df[df['Label'] == minority_label]

    # Downsample majority class
    df_majority_downsampled = resample(df_majority,
                                       replace=False,  # no duplication
                                       n_samples=len(df_minority),
                                       random_state=42)

    # Combine balanced data
    df_balanced = pd.concat([df_majority_downsampled, df_minority])

    # Shuffle for good measure
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save balanced dataset
    df_balanced.to_csv("dataset_balanced.csv", index=False)

    print("✅ Label balanced and saved to 'dataset_balanced.csv'")

else:
    print("\n✅ Label is already balanced within the 45–55% range. No changes made.")
