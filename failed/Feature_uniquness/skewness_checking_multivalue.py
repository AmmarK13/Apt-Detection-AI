import pandas as pd

def check_skew_in_multivalue_columns(file_path, columns, max_threshold=0.55):
    df = pd.read_csv(file_path)
    
    for col in columns:
        if col not in df.columns:
            print(f"⚠️ Column '{col}' not found in the dataset.\n")
            continue
        
        value_counts = df[col].value_counts(normalize=True).sort_values(ascending=False)
        top_val_percentage = value_counts.iloc[0]

        print(f"🔍 {col} ({len(value_counts)} unique values):")
        for val, pct in value_counts.items():
            print(f"   Value: {val} --> {pct * 100:.2f}%")
        
        if top_val_percentage > max_threshold:
            print(f"❌ Skewed: Top value dominates with {top_val_percentage * 100:.2f}%\n")
        else:
            print(f"✅ Balanced: No single value dominates\n")

# Example usage
file_path = r"d:\4th semester\SE\project\Dataset\balanced_dataset_v2.csv"
target_columns = ['Bwd Pkt Len Min', 'Fwd Seg Size Min', 'Fwd Pkt Len Min', 'Down/Up Ratio']

check_skew_in_multivalue_columns(file_path, target_columns)
