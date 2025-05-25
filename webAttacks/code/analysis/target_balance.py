import pandas as pd
import matplotlib.pyplot as plt

# Replace with your actual CSV file path and column name
csv_file = "D:/University/Software Engineering/Project/data/csic_reduced.csv"
classification_column = 'classification'

# Load the CSV
df = pd.read_csv(csv_file)

# Count 0s and 1s
counts = df[classification_column].value_counts()
percentages = (counts / len(df)) * 100

# Print the counts and percentages
print("Counts of classification values:")
print(counts)
print("\nPercentages of classification values:")
print(percentages.round(2).astype(str) + '%')

# Plotting
ax = counts.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Classification Count')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add percentage labels on top of each bar
for i, v in enumerate(counts):
    percentage = percentages[i]
    ax.text(i, v, f'{percentage:.1f}%', 
            ha='center', va='bottom')

plt.tight_layout()
plt.show()
