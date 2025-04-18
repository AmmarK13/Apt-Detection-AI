# remove featurws based of step 2 - analyze features
# check balance between unique vals of features

import pandas as pd
from collections import Counter

# Assuming `df` is your dataframe
file_path = "UNSW_NB15/cleaned_attempt3/3_normalised_data.csv" 
df = pd.read_csv(file_path) 
 

# Helper: function to compute imbalance ratio
def get_skew_ratio(series):
    counts = series.value_counts(normalize=True)
    max_ratio = counts.max()
    return max_ratio

# Step 1: Identify low-cardinality features (≤ 10 unique values), excluding 'label'
low_card_cols = [col for col in df.columns if col != 'label' and df[col].nunique() <= 10]

print(f"Low-cardinality features to check for skew: {low_card_cols}")

# Step 2: Drop skewed columns (outside 45%-55%)
cols_to_drop = []
for col in low_card_cols:
    skew = get_skew_ratio(df[col])
    if skew > 0.55 or skew < 0.45:
        cols_to_drop.append(col)

print(f"Columns dropped due to skew: {cols_to_drop}")
df.drop(columns=cols_to_drop, inplace=True)

# Step 3: Balance the 'label' column by undersampling
label_counts = df['label'].value_counts()
minority_class = label_counts.idxmin()
majority_class = label_counts.idxmax()

# Separate majority and minority classes
df_minority = df[df['label'] == minority_class]
df_majority = df[df['label'] == majority_class].sample(n=len(df_minority), random_state=42)

# Combine to make balanced dataset
df_balanced = pd.concat([df_minority, df_majority]).sample(frac=1, random_state=42).reset_index(drop=True)

print("Final balanced label distribution:")
print(df_balanced['label'].value_counts())

# drop attack_cat cause why not
df_balanced.drop(columns=["attack_cat"], inplace=True)
print("attack_cat col also dropped!")

# Optional: Save cleaned and balanced dataset
df_balanced.to_csv('UNSW_NB15/cleaned_attempt3/4_remove_features.csv' , index=False)
print("Saved cleaned and balanced dataset as 4_remove_features")
