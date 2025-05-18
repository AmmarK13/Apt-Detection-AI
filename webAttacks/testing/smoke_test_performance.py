# smoke_test_performance.py
import os
import time
import joblib
import pandas as pd

# Define model path based on project structure
MODEL_PATH = "d:/University/Software Engineering/Project/Apt-Detection-AI/webAttacks/code/model/attack_detection_rf_model.joblib"

def test_prediction_speed():
    print("\n===== Testing Prediction Speed =====\n")
    try:
        if os.path.exists(MODEL_PATH):
            # Load model and feature columns
            model, feature_columns = joblib.load(MODEL_PATH)
            
            # Load sample data
            sample_data_path = "D:/University/Software Engineering/Project/Output/scaled.csv"
            data = pd.read_csv(sample_data_path)
            
            # Prepare features
            X = data[feature_columns]
            
            # Test with different batch sizes
            batch_sizes = [10, 100, 1000]
            
            for size in batch_sizes:
                if len(X) >= size:
                    batch = X.head(size)
                    
                    # Measure prediction time
                    start_time = time.time()
                    model.predict(batch)
                    end_time = time.time()
                    
                    print(f"✓ Batch size {size}: {end_time - start_time:.4f} seconds")
                    print(f"✓ Average time per sample: {(end_time - start_time) / size:.6f} seconds")
            return True
        else:
            print(f"✗ Model file not found at {MODEL_PATH}")
            return False
    except Exception as e:
        print(f"✗ Error in performance test: {e}")
        return False

if __name__ == "__main__":
    test_prediction_speed()