# smoke_test_model_params.py
import os
import joblib

# Define model path based on project structure
MODEL_PATH = "d:/University/Software Engineering/Project/Apt-Detection-AI/webAttacks/code/model/attack_detection_rf_model.joblib"

def test_model_parameters():
    print("\n===== Testing Model Parameters =====\n")
    try:
        if os.path.exists(MODEL_PATH):
            # Load model
            model, feature_columns = joblib.load(MODEL_PATH)
            
            # Check Random Forest parameters
            print(f"✓ Model type: {type(model).__name__}")
            print(f"✓ Number of trees: {model.n_estimators}")
            print(f"✓ Max depth: {model.max_depth}")
            
            # Check feature importances
            if hasattr(model, "feature_importances_"):
                importances = sorted(zip(feature_columns, model.feature_importances_), 
                                    key=lambda x: x[1], reverse=True)
                print("\nTop 5 important features:")
                for feature, importance in importances[:5]:
                    print(f"- {feature}: {importance:.4f}")
            return True
        else:
            print(f"✗ Model file not found at {MODEL_PATH}")
            return False
    except Exception as e:
        print(f"✗ Error checking model parameters: {e}")
        return False

if __name__ == "__main__":
    test_model_parameters()