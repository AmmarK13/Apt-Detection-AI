import pandas as pd
import numpy as np

# Load your dataset
df = pd.read_csv(r"C:\Users\user\.cache\kagglehub\datasets\ogguy11\apt-detection\versions\1\02-14-2018.csv")  # Replace with your actual file

# Select only numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Dictionary to store issues
issues_summary = {
    "Outliers": {},
    "Negative Values": {},
    "Invalid Integers": {}
}

# Iterate through numerical columns
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Outliers
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    if len(outliers) > 0:
        issues_summary["Outliers"][col] = len(outliers)

    # Negative values (for columns that should always be positive)
    if (df[col] < 0).any():
        issues_summary["Negative Values"][col] = (df[col] < 0).sum()

    # Invalid integer values (if the column should be an integer but has decimals)
    if df[col].dtype == 'float64' and (df[col] % 1 != 0).any():
        issues_summary["Invalid Integers"][col] = (df[col] % 1 != 0).sum()

# Print summary of detected issues
for issue_type, columns in issues_summary.items():
    print(f"\n🔍 {issue_type}:")
    if columns:
        for col, count in columns.items():
            print(f"  - {col}: {count} issues found")
    else:
        print("  ✅ No issues detected")
