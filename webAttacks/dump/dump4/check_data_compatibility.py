import pandas as pd

def check_compatibility():
    # Load Payload data
    payload_data = pd.read_csv("D:\\University\\Software Engineering\\Project\\Cleaned Dataset\\csic_database.csv")
    
    # Load original training data to compare features
    original_data = pd.read_csv("d:\\University\\Software Engineering\\Project\\Apt-Detection-AI\\webAttacks\\data\\selected_features.csv")
    
    print("=== Data Compatibility Check Report ===")
    print("\nPayload Data Features:")
    print(payload_data.columns.tolist())
    print("\nOriginal Training Data Features:")
    print(original_data.columns.tolist())
    
    # Compare features
    payload_features = set(payload_data.columns)
    original_features = set(original_data.columns)
    
    missing_features = original_features - payload_features
    extra_features = payload_features - original_features
    common_features = payload_features.intersection(original_features)
    
    # Calculate matching percentage
    matching_percentage = (len(common_features) / len(original_features)) * 100
    
    print("\n=== Analysis Results ===")
    print(f"Total features in Payload data: {len(payload_features)}")
    print(f"Total features needed by model: {len(original_features)}")
    print(f"Feature matching percentage: {matching_percentage:.2f}%")
    
    if missing_features:
        print("\nMissing features (needed by model):")
        for feature in missing_features:
            print(f"- {feature}")
    
    if extra_features:
        print("\nExtra features (not used by model):")
        for feature in extra_features:
            print(f"- {feature}")
    
    # Final compatibility verdict
    is_compatible = len(missing_features) == 0
    print("\n=== Final Verdict ===")
    print(f"Is the data compatible? {'YES' if is_compatible else 'NO'}")
    if not is_compatible:
        print("Reason: Missing required features needed by the model")
    else:
        print("All required features are present")

if __name__ == "__main__":
    check_compatibility()