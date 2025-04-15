import pandas as pd

def check_binary_column_balance(file_path, max_diff_percent=10):
    df = pd.read_csv(file_path)

    # Find binary columns (exactly 2 unique values)
    binary_cols = df.nunique()[df.nunique() == 2].index

    print("⚖️ Checking balance in binary columns...\n")

    for col in binary_cols:
        counts = df[col].value_counts(normalize=True) * 100  # percentage
        val1, val2 = counts.values
        labels = counts.index.tolist()

        diff = abs(val1 - val2)
        if diff <= max_diff_percent:
            print(f"✅ {col} is balanced: {labels[0]} = {val1:.2f}%, {labels[1]} = {val2:.2f}%")
        else:
            print(f"❌ {col} is skewed: {labels[0]} = {val1:.2f}%, {labels[1]} = {val2:.2f}%")

# Example usage
file_path = r"d:\4th semester\SE\project\Dataset\02-14-2018_balanced.csv"
check_binary_column_balance(file_path, max_diff_percent=10)  # 55-45 split
