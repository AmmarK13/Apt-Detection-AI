import joblib
import pandas as pd
import numpy as np

def load_model(path):
    artifacts = joblib.load(path)
    return {
        'model': artifacts['model'],
        'scaler': artifacts['scaler'],
        'features': artifacts['feature_names'],
        'medians': artifacts.get('feature_medians', {})
    }

def safe_predict(model_pkg, data_path):
    try:
        new_data = pd.read_csv(data_path)
        new_data = new_data.select_dtypes(include=[np.number])
        full_features = pd.DataFrame(columns=model_pkg['features'])

        for col in model_pkg['features']:
            if col in new_data.columns:
                full_features[col] = pd.to_numeric(new_data[col], errors='coerce')
            else:
                full_features[col] = np.nan

        for col in full_features.columns:
            if full_features[col].isna().any():
                median_val = model_pkg['medians'].get(col, 0)
                full_features[col] = full_features[col].fillna(median_val)

        full_features = full_features.clip(-1e6, 1e6)
        scaled = model_pkg['scaler'].transform(full_features)

        return {
            'predictions': model_pkg['model'].predict(scaled),
            'probabilities': model_pkg['model'].predict_proba(scaled)[:, 1]
        }

    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    model = load_model("/home/kay/Documents/Workspace-S25/SE/SeProject/APT DETECTION/z.after mids progress/ddos_model.pkl")

    test_files = [
        "/home/kay/Documents/Workspace-S25/SE/SeProject/APT DETECTION/z.after mids progress/dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "/home/kay/Documents/Workspace-S25/SE/SeProject/APT DETECTION/z.after mids progress/dataset/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "/home/kay/Documents/Workspace-S25/SE/SeProject/APT DETECTION/z.after mids progress/dataset/Wednesday-workingHours.pcap_ISCX.csv",
        "/home/kay/Documents/Workspace-S25/SE/SeProject/APT DETECTION/z.after mids progress/dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX9.csv"
    ]

    for file in test_files:
        print(f"\n🔍 Testing {file}:")
        try:
            results = safe_predict(model, file)
            predictions = results['predictions']
            probs = results['probabilities']
            total = len(predictions)
            benign = (predictions == 0).sum()
            attack = (predictions == 1).sum()
            print(f"✅ Predictions: {predictions[:5]}...")
            print(f"📊 Prediction Breakdown:")
            print(f"   • Total Records: {total}")
            print(f"   • Benign (0): {benign} ({(benign/total)*100:.2f}%)")
            print(f"   • Attack (1): {attack} ({(attack/total)*100:.2f}%)")
            print(f"📊 Attack Probability Stats:")
            print(f"   • Mean: {np.mean(probs):.4f}")
            print(f"   • Max: {np.max(probs):.4f}")
            print(f"   • Min: {np.min(probs):.4f}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
