import pandas as pd
import numpy as np
import os

def validate_cleaning(original_path, cleaned_path, output_dir):
    # Check if the cleaned file exists
    if not os.path.exists(cleaned_path):
        print(f"❌ Cleaned file not found: {cleaned_path}")
        return
    
    # Load datasets
    original = pd.read_csv(original_path)
    cleaned = pd.read_csv(cleaned_path)
    
    print("\n🔍 Validation Report")
    print("====================")
    
    # 1. Check numeric conversion
    non_numeric_original = original.select_dtypes(exclude=[np.number]).columns
    non_numeric_cleaned = cleaned.select_dtypes(exclude=[np.number]).columns
    
    print("\n✅ Numeric Conversion Check:")
    print(f"Original non-numeric columns: {list(non_numeric_original)}")
    print(f"Cleaned non-numeric columns: {list(non_numeric_cleaned)}")
    
    # 2. Missing values check
    print("\n✅ Missing Values Check:")
    print(f"Original NaN count: {original.isna().sum().sum()}")
    print(f"Cleaned NaN count: {cleaned.isna().sum().sum()}")
    
    # 3. Infinite values check
    print("\n✅ Infinite Values Check:")
    print(f"Original inf count: {np.isinf(original.select_dtypes(include=[np.number])).sum().sum()}")
    print(f"Cleaned inf count: {np.isinf(cleaned.select_dtypes(include=[np.number])).sum().sum()}")
    
    # 4. Value range check
    print("\n✅ Value Range Check:")
    clipped_values = cleaned.select_dtypes(include=[np.number]).apply(
        lambda x: np.sum((x < -1e9) | (x > 1e9))).sum()
    print(f"Values outside [-1e9, 1e9] range: {clipped_values}")
    
    # 5. Outlier validation
    print("\n✅ Outlier Validation:")
    outliers = pd.read_csv(os.path.join(output_dir, "all_outliers.csv"))
    expected_removed = len(original) - len(cleaned)
    
    # Use set operations to determine the number of removed rows
    original_indices = set(original.index)
    cleaned_indices = set(cleaned.index)
    outlier_indices = set(outliers.index)
    
    actual_removed = len(outlier_indices - cleaned_indices)
    
    print(f"Expected removed rows: {expected_removed}")
    print(f"Actual removed rows: {actual_removed}")
    matching_percentage = actual_removed / expected_removed if expected_removed != 0 else 0
    print(f"Matching percentage: {matching_percentage:.2%}")

    # 6. Benign outlier retention check
    if 'Label' in cleaned.columns:
        benign_outliers = pd.read_csv(os.path.join(output_dir, "benign_outliers.csv"))
        
        # Ensure consistent data types for merging
        for col in benign_outliers.columns:
            if col in cleaned.columns:
                benign_outliers[col] = benign_outliers[col].astype(cleaned[col].dtype)
        
        retained_benign = cleaned.merge(benign_outliers, how='inner')
        print("\n✅ Benign Outlier Retention:")
        print(f"Total benign outliers: {len(benign_outliers)}")
        print(f"Retained benign outliers: {len(retained_benign)}")
        print(f"Retention percentage: {len(retained_benign)/len(benign_outliers):.2%}")

    # Final validation summary
    print("\n🔎 Validation Summary:")
    if (len(non_numeric_cleaned) == 0 and
        cleaned.isna().sum().sum() == 0 and
        clipped_values == 0 and
        (abs(actual_removed - expected_removed) < 5 or matching_percentage >= 0.80)):
        print("🟢 ALL CHECKS PASSED - Data cleaned successfully!")
    else:
        print("🔴 VALIDATION FAILED - Check cleaning steps")

# ========== CONFIGURATION ==========
ORIGINAL_FILE = r"Cleaned Dataset\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"  # Same as main script
OUTPUT_DIR = r"Recleaned Dataset\Friday-DDos"                  # Same as main script

if __name__ == "__main__":
    validate_cleaning(ORIGINAL_FILE, os.path.join(OUTPUT_DIR, "cleaned_data.csv"), OUTPUT_DIR)