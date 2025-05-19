import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv(r"C:\Users\user\.cache\kagglehub\datasets\ogguy11\apt-detection\versions\1\02-14-2018.csv")

# List of columns to visualize
columns_to_visualize = [
    "Dst Port", "Protocol", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
    "Fwd Pkt Len Mean", "Fwd Pkt Len Std", "Bwd Pkt Len Mean", "Bwd Pkt Len Std",
    "Flow Byts/s", "Flow Pkts/s", "Init Fwd Win Byts", "Init Bwd Win Byts"
]

# Set figure size
plt.figure(figsize=(15, 10))

# 📌 Boxplots for Outliers
for i, col in enumerate(columns_to_visualize):
    plt.subplot(4, 4, i+1)  # Adjust grid as needed
    sns.boxplot(y=df[col])
    plt.title(f"Boxplot of {col}")

plt.tight_layout()
plt.show()

# 📌 Histograms for Distribution
plt.figure(figsize=(15, 10))
for i, col in enumerate(columns_to_visualize):
    plt.subplot(4, 4, i+1)
    sns.histplot(df[col], bins=50, kde=True)
    plt.title(f"Histogram of {col}")

plt.tight_layout()
plt.show()

# 📌 Scatter Plot for Negative Values
plt.figure(figsize=(10, 5))
for col in ["Flow Duration", "Flow Pkts/s", "Init Fwd Win Byts", "Init Bwd Win Byts"]:
    plt.scatter(df.index, df[col], label=col, alpha=0.5)

plt.axhline(y=0, color='red', linestyle='--')  # Highlight zero line
plt.legend()
plt.title("Scatter Plot of Columns with Possible Negative Values")
plt.show()
