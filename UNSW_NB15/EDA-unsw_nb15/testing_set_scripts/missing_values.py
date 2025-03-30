import polars as pl 
import pandas as pd

# Load the dataset
file_path = "UNSW_NB15/cleaned_data/reduced_testing_set.csv"
df = pl.read_csv(file_path)

# null count per column
missing_counts = df.null_count()

# Ensure all columns are visible
pl.Config.set_tbl_cols(len(df.columns))

# Print missing value summary
print("Missing Values in Each Column:")
print(missing_counts)


# handle "-" in service feature  =============================================================

dash_count = (df["service"] == "-").sum()
print("\n\nCount of '-' in service column:", dash_count)

# Count actual null (missing) values
null_count = df["service"].is_null().sum()
print("Null count in service column:", null_count)


# print missing service count by proto
print("\nmisisng service vals and protocol")
print(df.filter(df["service"] == "-")["proto"].value_counts().to_pandas())


# correlation with attack labels
print("\nmisisng service vals and attack labels")
print(df.filter(df["service"] == "-")["label"].value_counts())


df = pd.read_csv(file_path, dtype=str)  # Read all columns as strings

# Replace dashes ("-") with proper missing values
df.replace("-", pd.NA, inplace=True)  # Using Pandas missing value

print("\n\n")
print("freq of each val of col:")
print(df["service"].value_counts())  # Shows frequency of each value
print("\n")

# Now, you can impute using the most frequent value
df.fillna(df.mode().iloc[0], inplace=True)
print("filled missing vals based on frequent one!")

# Verify if missing values are filled
print("\nfilling verification\nnull count of service")
df["service"] = df["service"].fillna(df["service"].mode().iloc[0])# Impute missing values
print(df["service"].isnull().sum())  # Check missing values again


#save csv with filled vals
df.to_csv("UNSW_NB15/cleaned_data/missingVals-testing.csv", index=False)
print("written to csv")