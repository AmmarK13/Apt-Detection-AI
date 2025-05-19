import pandas as pd

def drop_skewed_binary_columns(file_path, output_path, max_diff_percent=10):
    df = pd.read_csv(file_path)

    # Identify binary columns
    binary_cols = df.nunique()[df.nunique() == 2].index

    skewed_cols = []
    print("⚖️ Checking balance in binary columns...\n")

    for col in binary_cols:
        if col == 'Label':
            continue  # keep the target
        counts = df[col].value_counts(normalize=True) * 100
        val1, val2 = counts.values
        labels = counts.index.tolist()

        diff = abs(val1 - val2)
        if diff > max_diff_percent:
            print(f"❌ {col} is skewed: {labels[0]} = {val1:.2f}%, {labels[1]} = {val2:.2f}%")
            skewed_cols.append(col)
        else:
            print(f"✅ {col} is balanced: {labels[0]} = {val1:.2f}%, {labels[1]} = {val2:.2f}%")

    print("\n🧹 Dropping the following skewed binary columns:")
    for col in skewed_cols:
        print(f" - {col}")

    df_cleaned = df.drop(columns=skewed_cols)
    df_cleaned.to_csv(output_path, index=False)

    print(f"\n✅ Cleaned dataset saved to: {output_path}")

# Example usage
file_path = r"d:\4th semester\SE\project\Dataset\02-14-2018_balanced_removed_cols.csv"
output_path = r"d:\4th semester\SE\project\Dataset\02-14-2018_balanced_removed_cols_free_2skew.csv"
drop_skewed_binary_columns(file_path, output_path)
