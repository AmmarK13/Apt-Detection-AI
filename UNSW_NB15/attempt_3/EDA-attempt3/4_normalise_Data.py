import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load encoded dataset
file_path = "UNSW_NB15/cleaned_attempt3/2_encoded_data.csv" 
df = pd.read_csv(file_path)

# Columns to scale
to_scale = [
    'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sload', 'dload',
    'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit',
    'stcpb', 'dtcpb', 'tcprtt', 'synack', 'ackdat',
    'smean', 'dmean', 'response_body_len', 'ct_src_dport_ltm', 'ct_dst_sport_ltm'
]

# Initialize scaler
scaler = MinMaxScaler()

# Fit, transform, and round
df[to_scale] = scaler.fit_transform(df[to_scale]).round(2)

# Save the scaled dataset
output_file = "UNSW_NB15/cleaned_attempt3/3_normalised_data.csv" 
df.to_csv(output_file, index=False)
print(f"Scaling complete. Saved to: {output_file}")
