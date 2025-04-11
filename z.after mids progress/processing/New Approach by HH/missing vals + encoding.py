import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# === File Path Setup ===
file_path = "/home/kay/Documents/Workspace-S25/SE/SeProject/APT DETECTION/z.after mids progress/processing/New Approach by HH/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
cleaned_file_path = file_path.replace(".csv", "_CLEANED_BALANCED_CLEAN.csv")
output_dir = os.path.dirname(file_path)

# === Load Dataset ===
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# === 1. Data Type Sanitization ===
for col in df.select_dtypes('object').columns:
    try:
        df[col] = pd.to_numeric(df[col], errors='raise')
        print(f"Converted {col} to numeric")
    except:
        pass

# === 2. Missing Value Analysis ===
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100

# Visualization: Missing Values (Before)
fig, ax = plt.subplots(1, 2, figsize=(20,6))
sns.heatmap(df.isnull(), cmap="viridis", cbar=False, ax=ax[0], yticklabels=False)
ax[0].set_title("Missing Values - Before Handling")
missing_percentage[missing_percentage > 0].sort_values().plot(
    kind="barh", ax=ax[1], color="red"
)
ax[1].set_title("Missing Values Distribution")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "missing_values_before.png"))
plt.close()

# === 3. Strategic NaN Handling ===
threshold = 40 if len(df) < 100_000 else 20
cols_to_drop = missing_percentage[missing_percentage > threshold].index
df.drop(columns=cols_to_drop, inplace=True)
print(f"Dropped columns: {list(cols_to_drop)}")

# Imputation
num_cols = df.select_dtypes(include=np.number).columns
for col in num_cols:
    if df[col].isnull().sum() > 0:
        df[f"{col}_IS_MISSING"] = df[col].isnull().astype(int)
        df[col].fillna(-999 if df[col].min() >= 0 else 999, inplace=True)

# Categorical NaN
cat_cols = df.select_dtypes(exclude=np.number).columns
for col in cat_cols:
    df[col] = df[col].fillna("MISSING")

# === 4. Post-Cleaning Validation ===
constant_cols = df.columns[df.nunique() == 1]
df.drop(columns=constant_cols, inplace=True)
print(f"Removed constant columns: {list(constant_cols)}")

# Final missing value check
if df.isnull().sum().sum() > 0:
    raise ValueError("❌ NaN values still present after processing!")
else:
    print("✅ All missing values handled successfully")

# Visualization: Missing Values (After)
plt.figure(figsize=(12,6))
sns.heatmap(df.isnull(), cmap="viridis", cbar=False, yticklabels=False)
plt.title("Missing Values - After Final Handling")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "missing_values_after.png"))
plt.close()

# === 5. Label Preprocessing ===
df.columns = df.columns.str.strip()
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

print("\n=== Raw Label Values ===")
print("Unique labels before processing:", df["Label"].unique())
print("Label value counts before processing:")
print(df["Label"].value_counts())

df["Label"] = df["Label"].str.upper().str.strip()
ATTACK_LABELS = {"ATTACK", "DDOS", "DOS", "MALICIOUS"}
df["Label"] = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1 if x in ATTACK_LABELS else np.nan)

# Label validation
if df["Label"].isna().any():
    invalid = df[df["Label"].isna()]["Label"].unique()
    raise ValueError(f"Invalid labels found: {invalid}")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# === File Path Setup ===
file_path = "/home/kay/Documents/Workspace-S25/SE/SeProject/APT DETECTION/z.after mids progress/processing/New Approach by HH/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
cleaned_file_path = file_path.replace(".csv", "_CLEANED_BALANCED_CLEAN.csv")
output_dir = os.path.dirname(file_path)

# === Load Dataset ===
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# === 1. Data Type Sanitization ===
for col in df.select_dtypes('object').columns:
    try:
        df[col] = pd.to_numeric(df[col], errors='raise')
        print(f"Converted {col} to numeric")
    except:
        pass

# === 2. Missing Value Analysis ===
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100

# Visualization: Missing Values (Before)
fig, ax = plt.subplots(1, 2, figsize=(20,6))
sns.heatmap(df.isnull(), cmap="viridis", cbar=False, ax=ax[0], yticklabels=False)
ax[0].set_title("Missing Values - Before Handling")
missing_percentage[missing_percentage > 0].sort_values().plot(
    kind="barh", ax=ax[1], color="red"
)
ax[1].set_title("Missing Values Distribution")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "missing_values_before.png"))
plt.close()

# === 3. Strategic NaN Handling ===
threshold = 40 if len(df) < 100_000 else 20
cols_to_drop = missing_percentage[missing_percentage > threshold].index
df.drop(columns=cols_to_drop, inplace=True)
print(f"Dropped columns: {list(cols_to_drop)}")

# Imputation
num_cols = df.select_dtypes(include=np.number).columns
for col in num_cols:
    if df[col].isnull().sum() > 0:
        df[f"{col}_IS_MISSING"] = df[col].isnull().astype(int)
        df[col].fillna(-999 if df[col].min() >= 0 else 999, inplace=True)

# Categorical NaN
cat_cols = df.select_dtypes(exclude=np.number).columns
for col in cat_cols:
    df[col] = df[col].fillna("MISSING")

# === 4. Post-Cleaning Validation ===
constant_cols = df.columns[df.nunique() == 1]
df.drop(columns=constant_cols, inplace=True)
print(f"Removed constant columns: {list(constant_cols)}")

# Final missing value check
if df.isnull().sum().sum() > 0:
    raise ValueError("❌ NaN values still present after processing!")
else:
    print("✅ All missing values handled successfully")

# Visualization: Missing Values (After)
plt.figure(figsize=(12,6))
sns.heatmap(df.isnull(), cmap="viridis", cbar=False, yticklabels=False)
plt.title("Missing Values - After Final Handling")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "missing_values_after.png"))
plt.close()

# === 5. Label Preprocessing ===
df.columns = df.columns.str.strip()
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

print("\n=== Raw Label Values ===")
print("Unique labels before processing:", df["Label"].unique())
print("Label value counts before processing:")
print(df["Label"].value_counts())

df["Label"] = df["Label"].str.upper().str.strip()
ATTACK_LABELS = {"ATTACK", "DDOS", "DOS", "MALICIOUS"}
df["Label"] = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1 if x in ATTACK_LABELS else np.nan)

# Label validation
if df["Label"].isna().any():
    invalid = df[df["Label"].isna()]["Label"].unique()
    raise ValueError(f"Invalid labels found: {invalid}")

# === 6. Class Distribution Visualization ===
label_counts = df["Label"].value_counts()
print("\n=== Final Class Distribution ===")
print(f"Benign (0): {label_counts.get(0, 0)} samples")
print(f"Attack (1): {label_counts.get(1, 0)} samples")

plt.figure(figsize=(12, 5))

colors = []
labels = []
if 0 in label_counts.index:
    colors.append("#1f77b4")
    labels.append("BENIGN")
if 1 in label_counts.index:
    colors.append("#ff7f0e")
    labels.append("ATTACK")

# Bar plot
plt.subplot(1, 2, 1)
ax = sns.barplot(x=label_counts.index, y=label_counts.values, 
                 palette=colors, hue=label_counts.index, 
                 dodge=False, legend=False)
plt.xticks(ticks=label_counts.index, labels=labels, fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.title("Class Distribution", fontsize=14)
for p in ax.patches:
    ax.annotate(f"{p.get_height():,}", 
                (p.get_x() + p.get_width()/2., p.get_height()),
                ha='center', va='center', xytext=(0, 9), textcoords='offset points')

# Pie chart
if len(label_counts) > 1:
    plt.subplot(1, 2, 2)
    plt.pie(label_counts, labels=labels,
            autopct=lambda p: f"{p:.1f}%\n({int(p*sum(label_counts)/100)})",
            colors=colors, startangle=90,
            explode=[0.05]*len(labels),
            textprops={"fontsize": 12})
    plt.title("Class Proportion", fontsize=14)
else:
    plt.subplot(1, 2, 2)
    plt.text(0.5, 0.5, "Single Class Detected\nCannot Show Pie Chart", 
             ha='center', va='center', fontsize=12)
    plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "class_distribution.png"))
plt.close()

# === 7. Final Check ===
if 1 not in label_counts.index:
    raise ValueError("\n❌ CRITICAL ERROR: No attack samples detected!\n"
                     "Possible causes:\n"
                     "1. Dataset contains only benign traffic\n"
                     "2. Label conversion failed\n"
                     "3. Original data filtering removed attacks")

# === Save Cleaned Dataset ===
df.to_csv(cleaned_file_path, index=False)
print(f"\n✅ Cleaned dataset saved to: {cleaned_file_path}")


# === 7. Final Check ===
if 1 not in label_counts.index:
    raise ValueError("\n❌ CRITICAL ERROR: No attack samples detected!\n"
                     "Possible causes:\n"
                     "1. Dataset contains only benign traffic\n"
                     "2. Label conversion failed\n"
                     "3. Original data filtering removed attacks")

# === Save Cleaned Dataset ===
df.to_csv(cleaned_file_path, index=False)
print(f"\n✅ Cleaned dataset saved to: {cleaned_file_path}")
