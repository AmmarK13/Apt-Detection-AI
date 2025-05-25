import pandas as pd

# Load the dataset
df = pd.read_csv(r"C:\Users\user\.cache\kagglehub\datasets\ogguy11\apt-detection\versions\1\02-14-2018.csv")  # Replace with your actual file name

# Check for missing values
missing_values = df.isnull().sum()

# Filter only columns with missing values
missing_values = missing_values[missing_values > 0]

# Display results
if missing_values.empty:
    print("✅ No missing values found in the dataset.")
else:
    print("⚠️ Missing Values Detected:\n")
    for col, count in missing_values.items():
        percent = (count / len(df)) * 100
        print(f"📌 Column: {col} → Missing: {count} ({percent:.2f}%)")
