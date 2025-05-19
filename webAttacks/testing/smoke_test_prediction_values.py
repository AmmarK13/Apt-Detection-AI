# smoke_test_prediction_values.py
import pandas as pd

def test_prediction_values():
    print("\n===== Testing Prediction Values =====\n")
    try:
        # Load the predictions file
        predictions_df = pd.read_csv("D:/University/Software Engineering/Project\Output/Smoke Testing/smoke_test_predictions.csv")
        
        # Check if predictions are binary (0 or 1)
        unique_values = predictions_df["predicted_attack"].unique()
        print(f"✓ Unique prediction values: {unique_values}")
        
        if set(unique_values).issubset({0, 1}):
            print("✓ Predictions are valid binary classifications")
            return True
        else:
            print("✗ Predictions contain unexpected values")
            return False
    except Exception as e:
        print(f"✗ Error validating predictions: {e}")
        return False

if __name__ == "__main__":
    test_prediction_values()