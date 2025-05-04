import pandas as pd

# Load your dataset (replace the file path with your actual file)
df = pd.read_csv(r"D:\4th semester\SE\project\Dataset\Dropped_1_unique.csv")  # Replace with your dataset path

# List of TCP flag count columns to check
tcp_flag_columns = ['FIN Flag Cnt', 'Fwd PSH Flags', 'URG Flag Cnt', 'ACK Flag Cnt', 
                    'PSH Flag Cnt', 'ECE Flag Cnt']

# Check unique values for each column
for col in tcp_flag_columns:
    print(f"Unique values for column '{col}':")
    print(df[col].value_counts())
    print("\n")

import pandas as pd

# Load your dataset
df = pd.read_csv(r"D:\4th semester\SE\project\Dataset\New_Dataset_52_48.csv")  # Replace with the actual path

# List of columns to analyze
flag_columns = ['FIN Flag Cnt', 'Fwd PSH Flags', 'URG Flag Cnt', 'ACK Flag Cnt', 'PSH Flag Cnt', 'ECE Flag Cnt']

# Iterate through each flag column and group by 'Label' to see the distribution
for col in flag_columns:
    print(f"Unique values for column '{col}':")
    print(df.groupby([col, 'Label']).size().unstack(fill_value=0))
    print("\n" + "-"*50 + "\n")

