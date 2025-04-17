import pandas as pd

# Load the dataset
input_file = "dataset_balanced.csv"  # or your desired CSV
df = pd.read_csv(input_file)

# Prepare a log file to save output
output_file = "value_percentages.txt"
with open(output_file, "w") as f:
    for column in df.columns:
        f.write(f"🔍 Column: {column}\n")
        value_counts = df[column].value_counts(normalize=True) * 100  # percentage
        for val, pct in value_counts.items():
            f.write(f"  ➤ Value: {val} → {pct:.2f}%\n")
        f.write("\n")

print("✅ Percentage of unique values calculated for each column.")
print(f"📄 Results saved to '{output_file}'")
