import pandas as pd
import numpy as np
import random

# Define the required features
required_features = [
    'dur', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sload', 'dload',
    'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'stcpb', 'dtcpb', 'tcprtt', 'synack',
    'ackdat', 'smean', 'dmean', 'response_body_len', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
    'ct_flw_http_mthd', 'label'
]

# Define possible categorical values
service_values = ['-', 'http', 'ftp', 'dns', 'ssh', 'smtp']
state_values = ['CON', 'INT', 'FIN', 'RST', 'REQ']
ct_flw_http_mthd_values = [0, 1]  # assume 0 = no HTTP method, 1 = HTTP method used

# Generate synthetic data
def generate_synthetic_data(num_samples=1000):
    data = {
        'dur': np.random.exponential(scale=1.0, size=num_samples),
        'service': np.random.choice(service_values, size=num_samples),
        'state': np.random.choice(state_values, size=num_samples),
        'spkts': np.random.randint(1, 1000, size=num_samples),
        'dpkts': np.random.randint(1, 1000, size=num_samples),
        'sbytes': np.random.randint(1, 10000, size=num_samples),
        'dbytes': np.random.randint(1, 10000, size=num_samples),
        'rate': np.random.random(size=num_samples),
        'sload': np.random.random(size=num_samples) * 1000,
        'dload': np.random.random(size=num_samples) * 1000,
        'sloss': np.random.randint(0, 100, size=num_samples),
        'dloss': np.random.randint(0, 100, size=num_samples),
        'sinpkt': np.random.random(size=num_samples),
        'dinpkt': np.random.random(size=num_samples),
        'sjit': np.random.random(size=num_samples),
        'djit': np.random.random(size=num_samples),
        'stcpb': np.random.randint(0, 100000, size=num_samples),
        'dtcpb': np.random.randint(0, 100000, size=num_samples),
        'tcprtt': np.random.random(size=num_samples),
        'synack': np.random.random(size=num_samples),
        'ackdat': np.random.random(size=num_samples),
        'smean': np.random.randint(0, 1000, size=num_samples),
        'dmean': np.random.randint(0, 1000, size=num_samples),
        'response_body_len': np.random.randint(0, 5000, size=num_samples),
        'ct_src_dport_ltm': np.random.randint(0, 100, size=num_samples),
        'ct_dst_sport_ltm': np.random.randint(0, 100, size=num_samples),
        'ct_flw_http_mthd': np.random.choice(ct_flw_http_mthd_values, size=num_samples),
        'label': np.random.choice([0, 1], size=num_samples, p=[0.6, 0.4])  # 60% normal, 40% attack
    }
    df = pd.DataFrame(data)
    return df

# Save synthetic data to CSV
def save_synthetic_data(output_path='synthetic_data.csv', num_samples=1000):
    df = generate_synthetic_data(num_samples)
    df.to_csv(output_path, index=False)
    print(f"Synthetic data saved to {output_path}")

# Example usage
if __name__ == "__main__":
    save_synthetic_data(output_path='UNSW_NB15/transformation/synthetic_data.csv', num_samples=5000)
