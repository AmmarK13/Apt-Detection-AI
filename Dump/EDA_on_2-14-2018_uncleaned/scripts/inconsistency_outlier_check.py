import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv(r"C:\Users\user\.cache\kagglehub\datasets\ogguy11\apt-detection\versions\1\02-14-2018.csv")  # Replace with your actual file name

# Select numerical columns only
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Set Seaborn style
sns.set(style="whitegrid")

# Plot each numerical column separately
for col in numerical_cols:
    plt.figure(figsize=(8, 4))  # Set figure size for better visibility
    sns.boxplot(y=df[col], color="skyblue")
    plt.title(f"Boxplot of {col}")
    plt.ylabel("Value")
    plt.xlabel("")
    plt.show()  # Show one plot at a time
