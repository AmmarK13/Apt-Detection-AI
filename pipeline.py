import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import sys
import os

def load_dataset(file_path: str) -> pd.DataFrame:
    try:
        print(f"[INFO] Loading dataset: {file_path}")
        df = pd.read_csv(file_path)
        print(f"[SUCCESS] Loaded dataset with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        raise

def align_features(df: pd.DataFrame, required_features: list, categorical_features: list) -> pd.DataFrame:
    print("[INFO] Aligning features (strict mode for smoke testing)...")

    # Drop any extra columns
    df = df[[col for col in df.columns if col in required_features]]

    # Check for missing required columns
    missing_cols = [col for col in required_features if col not in df.columns]
    if missing_cols:
        print(f"\n\n[ERROR] Missing required columns: {missing_cols}")
        raise ValueError(f"Smoke test failed: missing required columns: {missing_cols}")

    # Reorder columns to match training data
    df = df[required_features]

    print("[SUCCESS] Features aligned")
    return df

def fix_unknowns(df: pd.DataFrame) -> pd.DataFrame:
    print("[INFO] Fixing unknown values...")
    known_services = ['ftp', 'smtp', 'dns', 'http', 'pop3', 'ssh', 'ftp-data', 
                      'irc', 'dhcp', 'snmp', 'ssl', 'radius']
    known_states = ['FIN', 'INT', 'CON', 'REQ', 'RST']
    known_labels = [0, 1]

    if 'service' in df.columns:
        df['service'] = df['service'].apply(lambda x: x if x in known_services else 'dns')

    if 'state' in df.columns:
        df['state'] = df['state'].apply(lambda x: x if x in known_states else 'INT')

    if 'label' in df.columns:
        df['label'] = df['label'].apply(lambda x: x if x in known_labels else 0)

    print("[SUCCESS] Unknowns handled")
    return df

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    print("[INFO] Filling missing values...")
    if 'service' in df.columns:
        df['service'] = df['service'].fillna('dns')
    if 'state' in df.columns:
        df['state'] = df['state'].fillna('INT')

    for col in df.columns:
        if col not in ['service', 'state']:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(0)
    print("[SUCCESS] Missing values filled")
    return df

def encode_features(df: pd.DataFrame):
    print("[INFO] Encoding categorical features...")
    try:
        if 'service' in df.columns:
            df['service'] = LabelEncoder().fit_transform(df['service'])

        if 'state' in df.columns:
            df['state'] = LabelEncoder().fit_transform(df['state'])

        if 'label' in df.columns and not pd.api.types.is_numeric_dtype(df['label']):
            df['label'] = LabelEncoder().fit_transform(df['label'])

        print("[SUCCESS] Features encoded")
        return df
    except Exception as e:
        print(f"[ERROR] Encoding failed: {e}")
        raise

def scale_data(df: pd.DataFrame, columns_to_scale: list) -> pd.DataFrame:
    print("[INFO] Scaling data...")
    scaler = MinMaxScaler()
    try:
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale]).round(2)
        print("[SUCCESS] Data scaled")
        return df
    except Exception as e:
        print(f"[ERROR] Scaling failed: {e}")
        raise

def transform(input_file: str, required_features: list, categorical_features: list, columns_to_scale: list) -> pd.DataFrame:
    try:
        df = load_dataset(input_file)
        df = align_features(df, required_features, categorical_features)
        df = fix_unknowns(df)
        df = fill_missing_values(df)
        df = encode_features(df)
        df = scale_data(df, columns_to_scale)
        print("[SUCCESS] Transformation pipeline completed")
        return df
    except Exception as e:
        print(f"[FATAL] Transformation failed: {e}")
        raise

def evaluate_model(input_file: str, model_path: str, required_features: list, categorical_features: list, columns_to_scale: list):
    try:
        print("[INFO] Evaluating model...")
        df = transform(input_file, required_features, categorical_features, columns_to_scale)

        if 'label' not in df.columns:
            raise ValueError("Missing 'label' column in the transformed data.")

        X = df.drop('label', axis=1)
        y = df['label']

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = joblib.load(model_path)
        print("[SUCCESS] Model loaded")

        y_pred = model.predict(X)

        print(f"\n[SUCCESS] Accuracy: {accuracy_score(y, y_pred):.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y, y_pred))
        print("\n\n\nClassification Report:")
        print(classification_report(y, y_pred))

    except Exception as e:
        print(f"[FATAL] Model evaluation failed: {e}")
        raise

# ------------------------
# RUNNING BLOCK (Smoke Test)
# ------------------------
if __name__ == "__main__":
    # new_file = "UNSW_NB15/transformation/synthetic_data.csv"
    # new_file = "UNSW_NB15/transformation/e.csv"
    new_file = "UNSW_NB15/cleaned_data/reduced_testing_set.csv"

    model_path = "UNSW_NB15/attempt_3/models/random_forest_model.joblib"

    required_features = [
        'dur', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sload', 'dload',
        'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'stcpb', 'dtcpb', 'tcprtt', 'synack', 'ackdat',
        'smean', 'dmean', 'response_body_len', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_flw_http_mthd', 'label'
    ]

    columns_to_scale = [
        'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sload', 'dload',
        'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit',
        'stcpb', 'dtcpb', 'tcprtt', 'synack', 'ackdat',
        'smean', 'dmean', 'response_body_len', 'ct_src_dport_ltm', 'ct_dst_sport_ltm'
    ]

    categorical_features = ['service', 'state']

    print("\n\n[START] Running transformation and evaluation as smoke test...")
    try:
        transformed_df = transform(new_file, required_features, categorical_features, columns_to_scale)
        transformed_df.to_csv("UNSW_NB15/transformation/transformed_test_file.csv", index=False)
        print("[SUCCESS] Transformed file saved")

        evaluate_model("UNSW_NB15/transformation/transformed_test_file.csv", model_path, required_features, categorical_features, columns_to_scale)
        print("[COMPLETE] Smoke test finished")
    except Exception as e:
        print(f"[FATAL] Smoke test aborted: {e}")
