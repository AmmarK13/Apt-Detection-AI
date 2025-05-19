import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from data_transform import data_transform  # Your custom preprocessing function

print("\n🔍 Running Smoke Test...\n")

# ST01 - Load Model
try:
    model = joblib.load('trained_model.pkl')
    assert model is not None
    print("✅ ST01 - Load Model: Model loaded successfully.")
except Exception as e:
    print(f"❌ ST01 - Load Model Failed: {e}")
    exit()

# ST02 - Load Sample Data
try:
    test_data = pd.read_csv('test.csv')
    assert isinstance(test_data, pd.DataFrame)
    assert not test_data.empty
    print("✅ ST02 - Load Sample Data: Test data loaded successfully.")
except Exception as e:
    print(f"❌ ST02 - Load Sample Data Failed: {e}")
    exit()

# ST03 - Run data_transform()
try:
    X_test_scaled = data_transform(test_data)
    assert X_test_scaled is not None
    print("✅ ST03 - Run data_transform(): Data transformed successfully.")
except Exception as e:
    print(f"❌ ST03 - Run data_transform() Failed: {e}")
    exit()

# ST04 - Predict on Sample
try:
    y_pred = model.predict(X_test_scaled)
    assert len(y_pred) == len(X_test_scaled)
    print("✅ ST04 - Predict on Sample: Model made predictions successfully.")
except Exception as e:
    print(f"❌ ST04 - Predict on Sample Failed: {e}")
    exit()

# ST05 - Output Validity
try:
    unique_preds = set(y_pred)
    assert unique_preds.issubset({0, 1})
    print("✅ ST05 - Output Validity: Predictions are binary (0 or 1).")
except Exception as e:
    print(f"❌ ST05 - Output Validity Failed: {e}")
    exit()

print("\n🎉 Smoke Test Passed Successfully!\n")
