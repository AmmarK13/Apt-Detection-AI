import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
file_path = "/home/kay/Documents/Workspace-S25/SE/SeProject/APT DETECTION/z.after mids progress/dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX_CLEANED_BALANCED_CLEAN.csv"
df = pd.read_csv(file_path)

# ----------------------
# 1. Whitespace Sanitization
# ----------------------
df.columns = df.columns.str.strip()
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# ----------------------
# 2. Debugging: Check Raw Labels
# ----------------------
print("\n=== Raw Label Values ===")
print("Unique labels before processing:", df["Label"].unique())
print("Label value counts before processing:")
print(df["Label"].value_counts())

# ----------------------
# 3. Label Validation & Conversion
# ----------------------
# Convert to uppercase and strip whitespace
df["Label"] = df["Label"].str.upper().str.strip()

# List all possible attack labels (modify according to your data)
ATTACK_LABELS = {"ATTACK", "DDOS", "DOS", "MALICIOUS"} 

# Create binary labels
df["Label"] = df["Label"].apply(
    lambda x: 0 if x == "BENIGN" else 1 if x in ATTACK_LABELS else np.nan
)

# Check for invalid labels
if df["Label"].isna().any():
    invalid = df[df["Label"].isna()]["Label"].unique()
    raise ValueError(f"Invalid labels found: {invalid}")

# ----------------------
# 4. Class Distribution Analysis
# ----------------------
label_counts = df["Label"].value_counts()
print("\n=== Final Class Distribution ===")
print(f"Benign (0): {label_counts.get(0, 0)} samples")
print(f"Attack (1): {label_counts.get(1, 0)} samples")

# ----------------------
# 5. Safe Visualization
# ----------------------
plt.figure(figsize=(12, 5))

# Dynamically adjust based on available classes
colors = []
labels = []
if 0 in label_counts.index:
    colors.append("#1f77b4")
    labels.append("BENIGN")
if 1 in label_counts.index:
    colors.append("#ff7f0e")
    labels.append("ATTACK")

# Bar Plot
plt.subplot(1, 2, 1)
ax = sns.barplot(x=label_counts.index, y=label_counts.values, 
                palette=colors, hue=label_counts.index, 
                dodge=False, legend=False)
plt.xticks(ticks=label_counts.index, labels=labels, fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.title("Class Distribution", fontsize=14)

# Add value labels
for p in ax.patches:
    ax.annotate(f"{p.get_height():,}", 
                (p.get_x() + p.get_width()/2., p.get_height()),
                ha='center', va='center', 
                xytext=(0, 9), 
                textcoords='offset points')

# Pie Chart (only if both classes exist)
if len(label_counts) > 1:
    plt.subplot(1, 2, 2)
    plt.pie(label_counts, labels=labels, 
            autopct=lambda p: f"{p:.1f}%\n({int(p*sum(label_counts)/100)})", 
            colors=colors, 
            startangle=90,
            explode=[0.05]*len(labels), 
            textprops={"fontsize": 12})
    plt.title("Class Proportion", fontsize=14)
else:
    plt.subplot(1, 2, 2)
    plt.text(0.5, 0.5, "Single Class Detected\nCannot Show Pie Chart", 
             ha='center', va='center', fontsize=12)
    plt.axis('off')

plt.tight_layout()
plt.show()

# ----------------------
# 6. Final Validation
# ----------------------
if 1 not in label_counts.index:
    raise ValueError("\n❌ CRITICAL ERROR: No attack samples detected!\n"
                     "Possible causes:\n"
                     "1. Dataset contains only benign traffic\n"
                     "2. Label conversion failed\n"
                     "3. Original data filtering removed attacks")

df.to_csv(file_path, index=False)
print(f"\n✅ Cleaned dataset saved to: {file_path}")