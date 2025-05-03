import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler

# Features expected by model
model_features = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std',
    'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
    'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
    'Min Packet Length', 'Max Packet Length', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
    'ACK Flag Count', 'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
    'Fwd Header Length.1', 'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes',
    'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd',
    'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
]

def data_transform(test_data: pd.DataFrame, reference_data_path='train_val.csv') -> np.ndarray:
    reference_data = pd.read_csv(reference_data_path)

    for col in model_features:
        if col not in test_data.columns:
            sampled_value = reference_data[col].dropna().sample(1).values[0]
            test_data[col] = sampled_value
        else:
            if test_data[col].isnull().sum() > 0:
                sampled_values = reference_data[col].dropna()
                if not sampled_values.empty:
                    test_data[col] = test_data[col].apply(
                        lambda x: random.choice(sampled_values) if pd.isna(x) else x
                    )
                else:
                    test_data[col] = test_data[col].fillna(0)

    X_test = test_data[model_features].copy()
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.fillna(X_test.mean(), inplace=True)
    X_test = X_test.clip(upper=1e10)
    X_test = X_test.astype(np.float32)

    scaler = StandardScaler()
    return scaler.fit_transform(X_test)
