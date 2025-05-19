import pandas as pd

# Load the dataset
df = pd.read_csv(r"C:\Users\user\.cache\kagglehub\datasets\ogguy11\apt-detection\versions\1\02-14-2018.csv")  # Replace with your actual file name

# Function to classify column types
def classify_columns(df):
    categorical = []
    numerical = []
    text = []

    for col in df.columns:
        if df[col].dtype == 'object':  # Text or Categorical Data
            if df[col].nunique() < 20:  # If unique values are small, consider it categorical
                categorical.append(col)
            else:
                text.append(col)
        else:  # Numerical Data
            numerical.append(col)

    return categorical, numerical, text

# Get classified column lists
categorical_cols, numerical_cols, text_cols = classify_columns(df)

# Print results
print("Categorical Features:", categorical_cols)
print("Numerical Features:", numerical_cols)
print("Text Features:", text_cols)
