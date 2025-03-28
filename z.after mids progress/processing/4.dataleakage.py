import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
file_path = "/home/kay/Documents/Workspace-S25/SE/SeProject/APT DETECTION/z.after mids progress/dataset/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
df = pd.read_csv(file_path)


# Ensure the 'Label' column is numeric (1 = Attack, 0 = Benign)
df["Label"] = df["Label"].astype(int)

# Compute correlation matrix with absolute values 
correlation_matrix = df.corr().abs()

# Sort correlations of all features with 'Label'
label_correlation = correlation_matrix["Label"].sort_values(ascending=False)

# Print the top 10 most correlated features
print("Top 10 Most Correlated Features with Label:")
print(label_correlation.head(10))

# Visualizing the correlation heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()

#-----------------------------Dropping----------------------------------

# Dropping highly correlated features (if any > 0.9)
correlation_threshold = 0.1
leaky_features = label_correlation[label_correlation > correlation_threshold].index.tolist()

# Remove 'Label' from this list (we need it)
leaky_features.remove("Label")

print(f"Dropping potentially leaky features: {leaky_features}")
df_cleaned = df.drop(columns=leaky_features)

# Save cleaned dataset

df_cleaned.to_csv(file_path, index=False)
print(f"Cleaned dataset saved to {file_path}")


#------------------checking remaining---------------
k =0
for i in df_cleaned.columns:
    print(f"{k}, {i} ")
    k +=1