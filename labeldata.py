import os
import pandas as pd

# Set cleaned data directory
cleaned_dir = "data/cleaned/"

# List all CSV files
csv_files = [f for f in os.listdir(cleaned_dir) if f.endswith(".csv")]

# Check each dataset
for file in csv_files:
    file_path = os.path.join(cleaned_dir, file)
    
    try:
        # Load dataset
        df = pd.read_csv(file_path, nrows=5000)  # Load only first 5000 rows for speed
        print(f"\n📂 Checking {file} - {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Show first few rows
        print(df.head())

        # Check for label column
        possible_labels = ["attack", "attack_type", "label", "category", "class"]
        for col in df.columns:
            if any(keyword in col.lower() for keyword in possible_labels):
                print(f"✅ {file} is labeled! Label column found: {col}")
                break
        else:
            print(f"❌ {file} does NOT have a label column.")

    except Exception as e:
        print(f"⚠️ Error reading {file}: {e}")
