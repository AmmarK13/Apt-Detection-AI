import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(r'd:\4th semester\SE\project\Dataset\balanced_dataset_v2_v3.csv')  # Replace with your actual file path

columns_to_check = [
    'Label', 'Fwd Seg Size Min', 'Fwd Act Data Pkts', 'Tot Fwd Pkts', 'Subflow Fwd Pkts',
    'Tot Bwd Pkts', 'Subflow Bwd Pkts', 'Bwd Pkt Len Max', 'Pkt Len Max', 'Fwd Header Len',
    'Bwd Header Len', 'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Pkt Len Max', 'Active Std',
    'Idle Std', 'Flow IAT Min', 'Subflow Fwd Byts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
    'Subflow Bwd Byts', 'Active Min', 'Active Mean', 'Active Max', 'Idle Max', 'Idle Min',
    'Idle Mean', 'Bwd Pkt Len Mean', 'Bwd Seg Size Avg', 'Bwd Pkt Len Std', 'Dst Port',
    'Fwd Pkt Len Mean', 'Fwd Seg Size Avg', 'Fwd Pkt Len Std', 'Pkt Len Mean', 'Pkt Size Avg',
    'Pkt Len Std', 'Pkt Len Var', 'Fwd IAT Min', 'Bwd IAT Min', 'Bwd IAT Max', 'Bwd IAT Tot',
    'Bwd IAT Mean', 'Bwd IAT Std', 'Fwd IAT Std', 'Flow Byts/s', 'Bwd Pkts/s', 'Flow IAT Std',
    'Flow IAT Max', 'Fwd IAT Max', 'Fwd IAT Mean', 'Fwd IAT Tot', 'Flow IAT Mean', 'Flow Duration',
    'Fwd Pkts/s', 'Flow Pkts/s'
]

threshold = 0.70
skewed_columns = []

def check_skewness_and_mode(col):
    value_counts = df[col].value_counts(normalize=True)
    top_value_ratio = value_counts.iloc[0]
    top_value = value_counts.index[0]
    if top_value_ratio >= threshold:
        return "Skewed", top_value
    else:
        return "Balanced", None

# Store results
results = []

# Analyze each column
for col in columns_to_check:
    try:
        skewness, top_value = check_skewness_and_mode(col)
        results.append({
            "Column": col,
            "Skewness": skewness,
            "Dominant Value": top_value if skewness == "Skewed" else "-"
        })
        
        if skewness == "Skewed":
            skewed_columns.append(col)
    except Exception as e:
        results.append({
            "Column": col,
            "Skewness": "Error",
            "Dominant Value": str(e)
        })

# Print table
result_df = pd.DataFrame(results)
print(result_df)

# Plot box plots for skewed columns
for col in skewed_columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Box Plot - {col}")
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()
