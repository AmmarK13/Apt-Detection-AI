# smoke_test_data_loading.py
import pandas as pd

def test_data_loading():
    print("\n===== Testing Data Loading =====\n")
    try:
        # Adjust the path to your actual data file
        sample_data_path = "D:/University/Software Engineering/Project/Output/scaled.csv"
        sample_data = pd.read_csv(sample_data_path, nrows=5)
        print(f"✓ Sample data loaded successfully with shape: {sample_data.shape}")
        print(f"✓ Columns: {sample_data.columns.tolist()}")
        return sample_data
    except Exception as e:
        print(f"✗ Error loading sample data: {e}")
        return None

if __name__ == "__main__":
    test_data_loading()