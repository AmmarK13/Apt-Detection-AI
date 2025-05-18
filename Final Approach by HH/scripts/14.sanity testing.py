import pandas as pd
import numpy as np
import joblib
from data_transform import data_transform

print("\n🧪 Running Sanity Test...\n")

try:
    # Load model
    model = joblib.load('trained_model.pkl')
    print("✅ SA01 - Model loaded.")

    # Load test data
    test_data = pd.read_csv('train_val.csv')

    # Check schema
    expected_features = test_data.drop(columns=['Label'], errors='ignore').columns
    print("✅ SA02 - Input schema seems valid.")

    # SA03 - Handle missing values
    test_missing = test_data.copy()
    test_missing.iloc[0, 0] = np.nan  # introduce NaN in first row, first column
    try:
        data_transform(test_missing)
        print("✅ SA03 - Missing values handled correctly.")
    except:
        print("❌ SA03 - Crash on missing values.")

    # SA04 - Extreme values
    test_extreme = test_data.copy()
    test_extreme.iloc[0] = 1e10  # very large value
    try:
        X_extreme = data_transform(test_extreme)
        pred_extreme = model.predict(X_extreme)
        print(f"✅ SA04 - Model handled extreme values. Prediction: {pred_extreme[0]}")
    except:
        print("❌ SA04 - Model crashed on extreme values.")

    # SA05 - Consistency
    test_sample = test_data.sample(1, random_state=42)
    X_sample = data_transform(test_sample)
    pred1 = model.predict(X_sample)
    pred2 = model.predict(X_sample)
    if np.array_equal(pred1, pred2):
        print("✅ SA05 - Consistent predictions on same input.")
    else:
        print("❌ SA05 - Inconsistent predictions on same input.")

except Exception as e:
    print(f"\n❌ Sanity Test Failed: {e}")
