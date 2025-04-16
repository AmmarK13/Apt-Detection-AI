

import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv(r"C:\Users\user\.cache\kagglehub\datasets\ogguy11\apt-detection\versions\1\02-14-2018.csv")  # Replace with your file name

# Count and percentage of each label
label_counts = df['Label'].value_counts()
label_percentages = (label_counts / len(df)) * 100

# Print results
print("Label Distribution (Counts):\n", label_counts)
print("\nLabel Distribution (Percentages):\n", label_percentages.round(2))

# Plotting the spread
plt.figure(figsize=(8, 5))
bars = plt.bar(label_counts.index, label_counts.values, color=['green', 'orange', 'red'])

# Annotate each bar with percentage
for bar, pct in zip(bars, label_percentages):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height + 5, f'{pct:.2f}%', 
             ha='center', va='bottom', fontsize=10)

plt.title("Label Distribution in Dataset")
plt.xlabel("Class Labels")
plt.ylabel("Number of Records")
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
