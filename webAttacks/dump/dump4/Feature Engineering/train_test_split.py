import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

def split_dataset(test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets while preserving class distribution.
    Also creates XGBoost DMatrix objects for direct use with XGBoost.
    """
    # Load the dataset with selected features
    data_path = "d:\\University\\Software Engineering\\Project\\Apt-Detection-AI\\webAttacks\\data\\selected_features.csv"
    df = pd.read_csv(data_path)
    
    print(f"Loaded dataset with {df.shape[0]} samples and {df.shape[1]} features")
    
    # Check class distribution
    class_distribution = df['classification'].value_counts(normalize=True)
    print("\nClass distribution:")
    for cls, pct in class_distribution.items():
        print(f"- Class {cls}: {pct:.2%}")
    
    # Handle categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'classification' in categorical_cols:
        categorical_cols.remove('classification')
    
    if categorical_cols:
        print(f"\nEncoding {len(categorical_cols)} categorical columns:")
        for col in categorical_cols:
            print(f"- {col}")
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    # Split features and target
    X = df.drop('classification', axis=1)
    y = df['classification']
    
    # Use stratified split to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )
    
    # Create train and test dataframes
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    # Create output directory if it doesn't exist
    output_dir = "d:\\University\\Software Engineering\\Project\\Apt-Detection-AI\\webAttacks\\data\\split"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save train and test sets as CSV
    train_path = os.path.join(output_dir, "train_data.csv")
    test_path = os.path.join(output_dir, "test_data.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    # Create XGBoost DMatrix objects
    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    dtest = xgb.DMatrix(data=X_test, label=y_test)
    
    # Save XGBoost binary format
    dtrain_path = os.path.join(output_dir, "dtrain.buffer")
    dtest_path = os.path.join(output_dir, "dtest.buffer")
    
    dtrain.save_binary(dtrain_path)
    dtest.save_binary(dtest_path)
    
    print(f"\nSplit complete:")
    print(f"- Training set: {X_train.shape[0]} samples ({(1-test_size):.0%})")
    print(f"- Testing set: {X_test.shape[0]} samples ({test_size:.0%})")
    print(f"\nFiles saved:")
    print(f"- CSV format: {train_path} and {test_path}")
    print(f"- XGBoost DMatrix format: {dtrain_path} and {dtest_path}")

if __name__ == "__main__":
    split_dataset(test_size=0.2, random_state=42)