# smoke_test_model_loading.py
import os
import joblib

# Define model path based on project structure
MODEL_PATH = "d:/University/Software Engineering/Project/Apt-Detection-AI/webAttacks/code/model/attack_detection_rf_model.joblib"

def test_model_loading():
    print("\n===== Testing Model Loading =====\n")
    try:
        # Check if model file exists
        if os.path.exists(MODEL_PATH):
            model, feature_columns = joblib.load(MODEL_PATH)
            print(f"✓ Model loaded successfully from {MODEL_PATH}")
            print(f"✓ Model type: {type(model).__name__}")
            print(f"✓ Number of features: {len(feature_columns)}")
            return True
        else:
            print(f"✗ Model file not found at {MODEL_PATH}")
            return False
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False

if __name__ == "__main__":
    test_model_loading()