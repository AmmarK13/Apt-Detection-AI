import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

#load dataset
file_path = "/home/kay/Documents/Workspace-S25/SE/SeProject/APT DETECTION/z.after mids progress/dataset/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
df = pd.read_csv(file_path)

# checking missing values
missing_values = df.isnull().sum()
missing_percentage = (df.isnull().sum() / len(df)) * 100
print("Missing values count: \n", missing_values)
print("Missing values percentage. \n", missing_percentage)

#Visualize:
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cmap="viridis", cbar=False, yticklabels=False)
plt.title("Missing Values Heatmap")
plt.show()
# Visualization: Bar chart of missing values
missing_percentage[missing_percentage > 0].sort_values().plot(
    kind="barh", figsize=(10, 6), color="red"
)
plt.xlabel("Percentage of Missing Values")
plt.ylabel("Columns")
plt.title("Missing Values per Column")
plt.show()

#------------dropping cols with missing values------------
# Drop columns with >40% missing values
threshold = 40  # Adjust threshold if needed
cols_to_drop = missing_percentage[missing_percentage > threshold].index
df.drop(columns=cols_to_drop, inplace=True)
print(f"Dropped columns: {list(cols_to_drop)}")

# Fill numerical columns with median
# Fill numerical columns with median (only for numeric columns)
num_cols = df.select_dtypes(include=["number"]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Fill categorical columns with most frequent value (mode)
cat_cols = df.select_dtypes(include=["object"]).columns
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)


# Check again
print("\nMissing Values After Handling:\n", df.isnull().sum())

# Visualization: Check if missing values are handled
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cmap="viridis", cbar=False, yticklabels=False)
plt.title("Missing Values After Handling")
plt.show()
