import pandas as pd
import random
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import joblib
import os

def load_data(file_path):
    """Load dataset."""
    return pd.read_csv(file_path)

def replace_infinities(df):
    """Replace +inf/-inf with NaN and drop rows with NaN."""
    numeric_df = df.select_dtypes(include=[np.number])
    print(f"+Inf count: {np.isposinf(numeric_df).sum().sum()}")
    print(f"-Inf count: {np.isneginf(numeric_df).sum().sum()}")
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

def convert_timestamp(df):
    """Convert 'Timestamp' column to seconds."""
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
        df['Timestamp'] = (df['Timestamp'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    return df

def convert_objects_to_numeric(df):
    """Convert object columns (except 'Label') to numeric."""
    for col in df.columns:
        if df[col].dtype == 'object' and col != 'Label':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def remove_outliers(df, threshold=7):
    """Remove outliers using Z-score method."""
    numeric_cols = [col for col in df.columns if col != 'Label']
    z_scores = np.abs(stats.zscore(df[numeric_cols], nan_policy='omit'))
    outlier_mask = (z_scores > threshold).any(axis=1)
    print(f"Outliers detected: {outlier_mask.sum()}")
    df = df[~outlier_mask]
    return df

def drop_remaining_nans(df):
    """Drop any remaining NaNs."""
    df.dropna(inplace=True)
    return df

def scale_features(df, scaler_save_path):
    """Scale features except 'Label' using MinMaxScaler and save scaler."""
    feature_cols = [col for col in df.columns if col != 'Label']
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    joblib.dump(scaler, scaler_save_path)
    print(f"Scaler saved to: {scaler_save_path}")
    return df

def remove_highly_correlated_features(df, threshold=0.99):
    """Remove highly correlated features."""
    feature_cols = [col for col in df.columns if col != 'Label']
    corr_matrix = df[feature_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    print(f"The following {len(to_drop)} features will be dropped due to high correlation: {to_drop}")

    df = df.drop(to_drop, axis=1)
    return df

def save_data(df, output_path):
    """Save processed data."""
    df.to_csv(output_path, index=False)
    print(f"✅ Processed dataset saved at: {output_path}")


def balance_attack_data(df, attack_labels=["FTP-BruteForce", "SSH-Bruteforce"]):
    """Balance attack data against 'Benign' samples."""
    balanced_data = {}
    
    for attack in attack_labels:
        df_attack = df[df["Label"] == attack]
        df_benign = df[df["Label"] == "Benign"][:df_attack.shape[0]]  # Balance "Benign" samples with attack samples
        
        # Concatenate the "Benign" data with the attack data
        balanced_df = pd.concat([df_benign, df_attack], axis=0)
        
        # Add the balanced dataframe to the dictionary
        balanced_data[attack] = balanced_df

        # Output the balanced dataset
        output_path = f"D:\\4th semester\\SE\\project\\Dataset\\balanced_{attack}.csv"
        balanced_df.to_csv(output_path, index=False)
        print(f"Balanced dataset for {attack} saved at {output_path}")
    
    return balanced_data



    

def match_model_features(df, model_features, training_df=None):
    """Ensure the dataframe has exactly the model_features, adding missing ones with mean values and dropping extras."""
    # Add missing columns (with mean values from training_df if provided)
    missing_cols = set(model_features) - set(df.columns)
    if missing_cols:
        print(f"Adding missing columns: {missing_cols}")
        for col in missing_cols:
            # If training_df is provided, use its mean value for the missing columns
            if training_df is not None and col in training_df.columns:
                df[col] = training_df[col].mean()
            else:
                df[col] = np.nan  # In case training_df is not provided, fill with NaN
    # Drop extra columns not in model_features
    extra_cols = set(df.columns) - set(model_features)
    if extra_cols:
        print(f"Dropping extra columns: {extra_cols}")
        df.drop(columns=extra_cols, inplace=True)
    
    # Ensure the columns are ordered the same as in model_features
    df = df[model_features]
    
    return df






def handle_missing_data(test_data, reference_data, model_features):
    """Handle missing columns and missing values in the test data by sampling from reference data."""
    for col in model_features:
        if col not in test_data.columns:
            # Sample from reference data if column is missing
            sampled_value = reference_data[col].dropna().sample(1).values[0]
            test_data[col] = sampled_value
        else:
            # Handle missing values in existing columns
            if test_data[col].isnull().sum() > 0:
                sampled_values = reference_data[col].dropna()
                if not sampled_values.empty:
                    test_data[col] = test_data[col].apply(
                        lambda x: random.choice(sampled_values) if pd.isna(x) else x
                    )
                else:
                    test_data[col] = test_data[col].fillna(0)  # If no data to sample from, fill with 0
    
    return test_data


from sklearn.preprocessing import LabelEncoder

def encode_labels(df, label_col='Label', encoder_save_path=None):
    """Encode the label column and optionally save the encoder."""
    le = LabelEncoder()
    df[label_col] = le.fit_transform(df[label_col])
    
    if encoder_save_path:
        joblib.dump(le, encoder_save_path)
        print(f"Label encoder saved at: {encoder_save_path}")
    
    return df, le

def transform_test_data(test_df, training_df, model_features):
    """Transform test data to match model feature space with sampling-based filling and filtering."""
    test_df = handle_missing_data(test_df, training_df, model_features)
    test_df = match_model_features(test_df, model_features, training_df)
    return test_df




def preprocess_pipeline(input_file, output_file, scaler_save_path, label_encoder_path=None, correlation_threshold=0.99, attack_labels=None):
    """Full preprocessing pipeline with attack balancing and label encoding."""
    df = load_data(input_file)
    
    if attack_labels:
        balance_attack_data(df, attack_labels)
    df = replace_infinities(df)
    df = convert_timestamp(df)
    df = convert_objects_to_numeric(df)
    df = remove_outliers(df)
    df = drop_remaining_nans(df)
    df = remove_highly_correlated_features(df, threshold=correlation_threshold)
    df = scale_features(df, scaler_save_path)
    df, le = encode_labels(df, encoder_save_path=label_encoder_path)

    save_data(df, output_file)
    return df, le




