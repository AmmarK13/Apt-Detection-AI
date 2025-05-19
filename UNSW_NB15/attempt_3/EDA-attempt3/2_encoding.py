import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the data
df = pd.read_csv("UNSW_NB15/cleaned_attempt3/1_missingVals.csv")

# Categorical columns
categorical_features = ['label', 'is_sm_ips_ports', 'service', 'state', 'attack_cat', 'proto']

# Preview unique values
for col in categorical_features:
    print(f"{col} unique values:", df[col].nunique(), df[col].unique()[:10])

# Apply Label Encoding to all categorical features
label_encoder = LabelEncoder()
for col in categorical_features:
    df[col] = label_encoder.fit_transform(df[col])
    print(f"{col} encoded values:", df[col].unique())

# Save encoded data
output_path = 'UNSW_NB15/cleaned_attempt3/2_encoded_data.csv'
df.to_csv(output_path, index=False)
print(f"Encoded data has been saved to '{output_path}'")
