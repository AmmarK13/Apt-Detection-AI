import pandas as pd

def inspect_csv(file_path):
    # Prevent pandas from truncating output
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.expand_frame_repr', False)

    df = pd.read_csv(file_path)

    # Checking the number of rows and columns
    rows, cols = df.shape
    print(f"📊 File Details:\nRows: {rows}, Columns: {cols}\n")

    print("🧠 Data Types:\n", df.dtypes, "\n")
    print("🕳️ Missing Values per Column:\n", df.isnull().sum(), "\n")
    print("🔁 Unique Values per Column:\n", df.nunique(), "\n")
    print("🔍 Sample Rows:\n", df.head(), "\n")

    print("✅ File inspection complete.")

# Example usage
file_path = r"d:\4th semester\SE\project\Dataset\02-14-2018_balanced_removed_cols_free_2skew.csv"
inspect_csv(file_path)
