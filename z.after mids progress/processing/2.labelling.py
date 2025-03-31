import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

#load dataset
file_path = "/home/kay/Documents/Workspace-S25/SE/SeProject/APT DETECTION/z.after mids progress/dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX_CLEANED_STRATIFIED_CLEAN.csv"
df = pd.read_csv(file_path)


#----------------------------------------------------------------------

#check if 'label' column exists 
if "Label" in df.columns:
    print("yes")
else:
    print("no")

#Answered no 

#check if 'label' column exists 
if " Label" in df.columns:
    print("yes")
else:
    print("no")
    
#Answered yes ------basically the column name is Label with a space in it

# --------------------- Strip the spaces ---------------------

# # Strip spaces from column names
# df.columns = df.columns.str.strip()
# # Strip spaces from column values

# #check if 'label' column exists 
# if "Label" in df.columns:
#     print("yes")
# else:
#     print("no")

# #Answered no 

# #check if 'label' column exists 
# if " Label" in df.columns:
#     print("yes")
# else:
#     print("no")
    
# df.to_csv(file_path, index=False)
# print(f"✅ strip dataset saved as {file_path}")

# --------------------- Labelling -----------------------------

# # Check if 'Label' column exists
# if "Label" in df.columns:
#     # Convert 'BENIGN' to 0, everything else (attack) to 1
#     df["Label"] = df["Label"].apply(lambda x: 0 if x.strip().upper() == "BENIGN" else 1)
    
#     print("✅ Label Encoding Complete!")
#     print("🔍 Encoded Labels: 0 = BENIGN, 1 = ATTACK")
# else:
#     print("⚠️ No 'Label' column found. Skipping encoding.")
 
# # Save encoded dataset --to original file--
# # output_file = "/home/kay/Documents/Workspace-S25/SE/SeProject/APT DETECTION/z.after mids progress/dataset/encoded_dataset.csv"
# df.to_csv(file_path, index=False)

# print(f"✅ Encoded dataset saved as {file_path}")

# ------------------ counting the number of benign and attack instances in the dataset ------------------

# Count occurrences of each label
label_counts = df["Label"].value_counts()

# Print counts
print("Label Distribution:")
print(label_counts)

# Visualization - Bar Plot
plt.figure(figsize=(8, 5))
sns.barplot(x=label_counts.index, y=label_counts.values, palette=["blue", "red"])
plt.xticks(ticks=[0, 1], labels=["BENIGN (0)", "ATTACK (1)"])
plt.xlabel("Class Labels")
plt.ylabel("Number of Samples")
plt.title("Distribution of Benign vs. Attack Samples")
plt.show()

# Visualization - Pie Chart
plt.figure(figsize=(7, 7))
plt.pie(
    label_counts,
    labels=["BENIGN (0)", "ATTACK (1)"],
    autopct="%1.1f%%",
    colors=["blue", "red"],
    startangle=140,
    explode=(0.05, 0.1),
    shadow=True
)
plt.title("Percentage of Benign vs. Attack")
plt.show()

