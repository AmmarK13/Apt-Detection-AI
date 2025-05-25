# smoke_test_prediction.py
import sys
import os

# Add the project root to the path to import from code module
sys.path.append("d:/University/Software Engineering/Project/Apt-Detection-AI/webAttacks")

try:
    from code.model.random_forest import predict_attack
    
    def test_prediction():
        print("\n===== Testing Prediction Functionality =====\n")
        try:
            # Adjust the path to your actual data file
            sample_data_path = "D:/University/Software Engineering/Project/Output/scaled.csv"
            
            # Make predictions (this will load the model internally)
            result_df = predict_attack(sample_data_path, output_csv_path="D:/University/Software Engineering/Project\Output/Smoke Testing/smoke_test_predictions.csv")
            
            # Check if predictions were generated
            if "predicted_attack" in result_df.columns:
                print(f"✓ Predictions generated successfully")
                print(f"✓ Number of predictions: {len(result_df)}")
                print(f"✓ Prediction distribution: {result_df['predicted_attack'].value_counts().to_dict()}")
                return True
            else:
                print("✗ No predictions column found in results")
                return False
        except Exception as e:
            print(f"✗ Error during prediction: {e}")
            return False
    
    if __name__ == "__main__":
        test_prediction()
        
except ImportError:
    print("✗ Could not import predict_attack function from code.model.random_forest")
    print("  Make sure you're running this script from the project root directory")