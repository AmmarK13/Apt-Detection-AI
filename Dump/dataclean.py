import os
import pandas as pd

# Set data directory
data_dir = "data/"
cleaned_dir = "data/cleaned/"

# Ensure cleaned directory exists
os.makedirs(cleaned_dir, exist_ok=True)

# List all CSV files
csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

# Process each file in chunks instead of loading everything at once
chunk_size = 50000  # Adjust based on available RAM
for file in csv_files:
    file_path = os.path.join(data_dir, file)
    cleaned_path = os.path.join(cleaned_dir, file)
    
    print(f"\n🚀 Processing {file} in chunks...")
    
    # Open CSV & process in chunks
    with pd.read_csv(file_path, chunksize=chunk_size, low_memory=True) as reader:
        for i, chunk in enumerate(reader):
            # Drop missing values in the chunk
            chunk.dropna(inplace=True)
            
            # Save the first chunk with headers, append rest without headers
            mode = 'w' if i == 0 else 'a'
            header = True if i == 0 else False
            chunk.to_csv(cleaned_path, mode=mode, index=False, header=header)

            print(f"✅ Processed chunk {i+1} of {file}")

    print(f"📂 Cleaned file saved: {cleaned_path}")

print("\n🎯 All datasets cleaned & saved successfully!")
