# smoke_test_data_compatibility.py
import os
import joblib
import pandas as pd

# Define model path based on project structure
MODEL_PATH = "d:/University/Software Engineering/Project/Apt-Detection-AI/webAttacks/code/model/attack_detection_rf_model.joblib"

def test_data_compatibility():
    print("\n===== Testing Data Compatibility =====\n")
    try:
        # Load sample data
        sample_data_path = "D:/University/Software Engineering/Project/Output/scaled.csv"
        sample_data = pd.read_csv(sample_data_path, nrows=5)
        
        if os.path.exists(MODEL_PATH):
            # Load model and expected feature columns
            model, feature_columns = joblib.load(MODEL_PATH)
            
            # Check if all required columns exist in the sample data
            missing_columns = [col for col in feature_columns if col not in sample_data.columns]
            
            if missing_columns:
                print(f"✗ Missing expected columns: {missing_columns}")
                return False
            else:
                print(f"✓ All {len(feature_columns)} expected columns present")
                
                # Check for NaN values
                if sample_data[feature_columns].isna().any().any():
                    print("✗ NaN values found in sample data")
                    return False
                else:
                    print("✓ No NaN values found in sample data")
                    return True
        else:
            print(f"✗ Model file not found at {MODEL_PATH}")
            return False
    except Exception as e:
        print(f"✗ Error checking data compatibility: {e}")
        return False

if __name__ == "__main__":
    test_data_compatibility()