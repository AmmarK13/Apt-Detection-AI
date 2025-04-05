import pandas as pd
import numpy as np
import re
import os

def map_columns(df):
    """
    Map various column names to standardized format
    """
    # Common column name mappings
    url_columns = ['URL', 'url', 'request_http_request', 'Destination', 'request', 'uri']
    content_columns = ['content', 'payload', 'request_body', 'Body', 'data']
    label_columns = ['classification', 'label', 'attack_type', 'class', 'Label']
    
    # Find available columns
    available_columns = df.columns.str.lower()
    
    # Map URL column
    for col in url_columns:
        if col.lower() in available_columns:
            df['URL'] = df[col]
            break
    if 'URL' not in df.columns:
        df['URL'] = ''  # Default empty if no URL column found
        
    # Map content column
    for col in content_columns:
        if col.lower() in available_columns:
            df['content'] = df[col]
            break
    if 'content' not in df.columns:
        df['content'] = ''  # Default empty if no content column found
        
    # Map classification column
    for col in label_columns:
        if col.lower() in available_columns:
            df['classification'] = df[col]
            break
    if 'classification' not in df.columns:
        df['classification'] = 0  # Default to normal if no classification found
    
    return df

def enhance_features(df):
    """
    Enhance the dataset with the same features used in training
    """
    print("Enhancing features...")
    
    # First map columns to standard format
    df = map_columns(df)
    
    # Convert all strings to string type for consistency
    df['URL'] = df['URL'].astype(str)
    df['content'] = df['content'].astype(str)
    
    # URL-related features
    df['url_length'] = df['URL'].str.len()
    df['path_depth'] = df['URL'].str.count('/')
    df['query_length'] = df['URL'].apply(lambda x: len(x.split('?')[1]) if '?' in x else 0)
    df['param_count'] = df['URL'].apply(lambda x: x.count('&') + 1 if '?' in x else 0)
    
    # Content analysis
    df['content_length'] = df['content'].str.len()
    df['content_numbers'] = df['content'].apply(lambda x: sum(c.isdigit() for c in x))
    df['content_special_chars'] = df['content'].apply(lambda x: len(re.findall(r'[^a-zA-Z0-9\s]', x)))
    
    # Security checks
    df['has_hex_encoding'] = df.apply(lambda x: 1 if re.search(r'%[0-9a-fA-F]{2}', x['content'] + x['URL']) else 0, axis=1)
    df['has_base64'] = df['content'].apply(lambda x: 1 if re.search(r'^[A-Za-z0-9+/=]+$', x) else 0)
    df['contains_special_chars'] = df.apply(lambda x: 1 if re.search(r'[<>\'";]', x['content'] + x['URL']) else 0, axis=1)
    df['contains_sql_injection'] = df.apply(lambda x: 1 if re.search(r'(?i)(union|select|from|where|drop|delete|insert|update)', x['content'] + x['URL']) else 0, axis=1)
    df['contains_xss'] = df.apply(lambda x: 1 if re.search(r'(?i)(<script|javascript:|on\w+\s*=)', x['content'] + x['URL']) else 0, axis=1)
    df['contains_command_injection'] = df.apply(lambda x: 1 if re.search(r'(?i)([;&|`]|\$\()', x['content'] + x['URL']) else 0, axis=1)
    
    # Add HTTP headers if available
    df['content-type'] = df.get('content-type', 'text/plain')
    df['Accept'] = df.get('Accept', '*/*')
    df['cookie'] = df.get('cookie', '')
    
    print("Features enhanced. Columns available:", df.columns.tolist())
    return df

def check_compatibility(original_data, transformed_data):
    """
    Check compatibility between transformed data and original training data
    """
    print("\n=== Compatibility Check ===")
    original_features = set(original_data.columns)
    transformed_features = set(transformed_data.columns)
    
    missing_features = original_features - transformed_features
    extra_features = transformed_features - original_features
    common_features = original_features.intersection(transformed_features)
    
    matching_percentage = (len(common_features) / len(original_features)) * 100
    
    print(f"Feature matching: {matching_percentage:.2f}%")
    print(f"Common features: {len(common_features)}")
    
    if missing_features:
        print("\nMissing features:")
        for feature in missing_features:
            print(f"- {feature}")
    
    if extra_features:
        print("\nExtra features:")
        for feature in extra_features:
            print(f"- {feature}")
    
    return len(missing_features) == 0

def transform_and_check(input_path):
    """
    Main pipeline to transform data and check compatibility
    """
    print(f"Loading data from: {input_path}")
    new_data = pd.read_csv(input_path)
    print("Original columns:", new_data.columns.tolist())
    
    # Load original training data for comparison
    original_data = pd.read_csv("d:\\University\\Software Engineering\\Project\\Apt-Detection-AI\\webAttacks\\data\\selected_features.csv")
    
    # Transform the data
    transformed_data = enhance_features(new_data)
    
    # Check compatibility
    is_compatible = check_compatibility(original_data, transformed_data)
    
    if is_compatible:
        # Save transformed data
        output_dir = "d:\\University\\Software Engineering\\Project\\Apt-Detection-AI\\webAttacks\\data\\transformed"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "transformed_data.csv")
        transformed_data.to_csv(output_path, index=False)
        print(f"\nTransformed data saved to: {output_path}")
    else:
        print("\nWarning: Transformed data is not fully compatible with the model")
        print("Adding missing features with default values...")
        
        # Add missing features with default values
        for col in original_data.columns:
            if col not in transformed_data.columns:
                transformed_data[col] = 0
        
        # Reorder columns to match original data
        transformed_data = transformed_data[original_data.columns]
        
        # Save transformed data with updated path
        output_dir = "D:\\University\\Software Engineering\\Project\\transformed"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "transformed_data.csv")
        transformed_data.to_csv(output_path, index=False)
        print(f"Transformed data saved to: {output_path}")
    
    return transformed_data, is_compatible

if __name__ == "__main__":
    # Example usage with your dataset
    input_path = "D:\\University\\Software Engineering\\Project\\Cleaned Dataset\\Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
    transformed_data, is_compatible = transform_and_check(input_path)