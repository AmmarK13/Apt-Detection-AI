import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

#load dataset
file_path = "/home/kay/Documents/Workspace-S25/SE/SeProject/APT DETECTION/z.after mids progress/dataset/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
df = pd.read_csv(file_path)

#checking duplicate count 
duplicate_count = df.duplicated().sum()
print(f"Duplicate count: {duplicate_count}")

# 🔄 Visualizing duplicates (before removing)
plt.figure(figsize=(8, 4))
sns.barplot(x=["Unique Rows", "Duplicate Rows"], y=[len(df) - duplicate_count, duplicate_count], palette=["blue", "red"])
plt.ylabel("Count")
plt.title("Duplicate Rows in Dataset")
plt.show()

# ------------------------ removing duplicate ------------------------
df.drop_duplicates(inplace=True)

# Save the cleaned dataset
output_file = "/home/kay/Documents/Workspace-S25/SE/SeProject/APT DETECTION/z.after mids progress/dataset/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
df.to_csv(output_file, index=False)

print(f"Duplicates removed. Cleaned dataset saved as {output_file}")

# Visualizing dataset after removing duplicates
plt.figure(figsize=(8, 4))
sns.barplot(x=["Remaining Rows"], y=[len(df)], color="blue")
plt.ylabel("Count")
plt.title("Dataset After Removing Duplicates")
plt.show()