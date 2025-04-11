import joblib
import pandas as pd

def compare_features(model_path, new_data_path):
    # Load model features
    model = joblib.load(model_path)
    model_features = set(model['feature_names'])
    
    # Load new data features
    new_data = pd.read_csv(new_data_path, nrows=1)  # Just read header
    new_data_features = set(new_data.columns)
    
    # Compare features
    missing = model_features - new_data_features
    extra = new_data_features - model_features
    
    print(f"Missing features ({len(missing)}):\n{sorted(missing)}")
    print(f"\nExtra features ({len(extra)}):\n{sorted(extra)}")

# Usage
compare_features(
    "/home/kay/Documents/Workspace-S25/SE/SeProject/APT DETECTION/z.after mids progress/ddos_model.pkl",
    
    "/home/kay/Documents/Workspace-S25/SE/SeProject/APT DETECTION/z.after mids progress/dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv (2)"
)