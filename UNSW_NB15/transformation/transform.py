import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_dataset(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def align_features(df: pd.DataFrame, required_features: list, categorical_features: list) -> pd.DataFrame:
    """
    Ensures that the DataFrame `df` has exactly the columns listed in `required_features`.
    - Drops any extra columns not in required_features.
    - Adds any missing columns with default value 0 for numerical col and '-' for categorical col.

    Returns:
        pd.DataFrame: The aligned DataFrame with columns in the same order as required_features.
    """
    
    # categorical_features = ['service', 'state']

    # Drop any extra columns
    df = df[[col for col in df.columns if col in required_features]]
    
    # Add missing columns
    missing_cols = [col for col in required_features if col not in df.columns]
    for col in missing_cols:
        if col in categorical_features:
            df[col] = '-'  # Set default for categorical columns (can be adjusted as needed)
            df[col] = df[col].astype('category')  # Ensure it's a categorical type
        else:
            df[col] = 0  # Default for non-categorical columns

    # Reorder columns to match required_features exactly
    df = df[required_features]

    return df

def fix_unknowns(df: pd.DataFrame) -> pd.DataFrame:
    # Known values from training
    known_services = ['ftp', 'smtp', 'dns', 'http', 'pop3', 'ssh', 'ftp-data', 
                      'irc', 'dhcp', 'snmp', 'ssl', 'radius']
    known_states = ['FIN', 'INT', 'CON', 'REQ', 'RST']
    known_labels = [0, 1]

    # Handle service column
    if 'service' in df.columns:
        df['service'] = df['service'].apply(
            lambda x: x if x in known_services else 'dns'
        )

    # Handle state column
    if 'state' in df.columns:
        df['state'] = df['state'].apply(
            lambda x: x if x in known_states else 'INT'
        )

    # Handle label column
    if 'label' in df.columns:
        df['label'] = df['label'].apply(
            lambda x: x if x in known_labels else 0
        )

    return df

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    # Fill missing service and state
    if 'service' in df.columns:
        df['service'] = df['service'].fillna('dns')
    
    if 'state' in df.columns:
        df['state'] = df['state'].fillna('INT')
    
    # Fill missing values for all other numerical columns with 0
    for col in df.columns:
        if col not in ['service', 'state']:
            if df[col].dtype in ['float64', 'int64']:  # numeric types
                df[col] = df[col].fillna(0)

    return df

def encode_features(df: pd.DataFrame):
    encoder_service = LabelEncoder()
    encoder_state = LabelEncoder()
    encoder_label = LabelEncoder()
    
    if 'service' in df.columns:
        df['service'] = encoder_service.fit_transform(df['service'])
    
    if 'state' in df.columns:
        df['state'] = encoder_state.fit_transform(df['state'])

    if 'label' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['label']):
            df['label'] = encoder_label.fit_transform(df['label'])
        else:
            encoder_label = None  # no need for encoder if already numeric

    return df   #, encoder_service, encoder_state, encoder_label

def scale_data(df: pd.DataFrame, columns_to_scale: list) -> pd.DataFrame:
    # Ensure the input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected input df to be a pandas DataFrame.")

    # Ensure columns_to_scale exist in the DataFrame
    missing_cols = [col for col in columns_to_scale if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in the DataFrame: {', '.join(missing_cols)}")

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Check if the DataFrame has valid columns
    try:
        # Fit and transform the selected columns
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale]).round(2)
    except ValueError as e:
        print(f"Error in scaling columns: {e}")
        raise
    
    return df

def transform(input_file: str, required_features: list, categorical_features: list, columns_to_scale: list) -> pd.DataFrame:
    """
    Applies a sequence of transformations to the input data:
    - Load data
    - Align features
    - Fix unknown values
    - Fill missing values
    - Encode categorical features
    - Scale numerical features
    
    Returns:
        pd.DataFrame: The transformed DataFrame.
    """
    # Load the dataset
    df = pd.read_csv(input_file)

    # Align features (ensure all required columns are present)
    df = align_features(df, required_features, categorical_features)
    # print(type(df))
    
    # Fix unknown values (e.g., replace '-' in categorical columns)
    df = fix_unknowns(df)
    # print(type(df))
    
    # Fill missing values (e.g., fill NaN with 0 for numeric or '-' for categorical)
    df = fill_missing_values(df)
    # print(type(df))


    # Encode categorical features (if applicable)
    df = encode_features(df)
    # print(type(df))

    # Scale numerical features
    # print(type(df))  # This should output <class 'pandas.core.frame.DataFrame'>
    df = scale_data(df, columns_to_scale)
    
    return df

def evaluate_model(input_file: str, model_path: str, required_features: list, categorical_features: list, columns_to_scale: list):
    """
    Loads the model from the given path, tests it on the transformed data, and prints evaluation results.
    
    Parameters:
    - input_file: Path to the input data file (CSV).
    - model_path: Path to the saved model.
    - required_features: List of required features in the dataset.
    - categorical_features: List of categorical features.
    - columns_to_scale: List of columns to be scaled.
    """
    # # Step 1: Transform the data
    df = transform(input_file, required_features, categorical_features, columns_to_scale)

    print(df.columns)  # Print column names to verify if 'label' is present

    
    # Step 2: Separate features (X) and target (y)
    X = df.drop('label', axis=1)  # Assuming 'label' is the target column
    y = df['label']
    
    # Step 3: Load the Random Forest model
    model = joblib.load(model_path)
    
    # Step 4: Make predictions
    y_pred = model.predict(X)
    
    # Step 5: Evaluate the model
    accuracy = accuracy_score(y, y_pred)
    conf_matrix = confusion_matrix(y, y_pred)
    class_report = classification_report(y, y_pred)
    
    # Print the results
    print(f"\n\nAccuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)


if __name__ == "__main__":
    # new_file = "path/to/new/test_file.csv"


    new_file ="UNSW_NB15/transformation/synthetic_data.csv"



    # new_file = "UNSW_NB15/cleaned_data/reduced_testing_set.csv"

    required_features = [
        'dur', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sload', 'dload',
        'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'stcpb', 'dtcpb', 'tcprtt', 'synack', 'ackdat',
        'smean', 'dmean', 'response_body_len', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_flw_http_mthd', 'label'
    ]


    columns_to_scale = [
    'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sload', 'dload',
    'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit',
    'stcpb', 'dtcpb', 'tcprtt', 'synack', 'ackdat',
    'smean', 'dmean', 'response_body_len', 'ct_src_dport_ltm', 'ct_dst_sport_ltm'
    ]
    categorical_features = ['service', 'state']

    print("transforming....")
    transformed_df = transform(new_file,required_features, categorical_features,columns_to_scale )
    # Save if you want
    transformed_df.to_csv("UNSW_NB15/transformation/transformed_test_file.csv", index=False)
    print("\n\ntransformed and saved file as transformed_test_file.csv")
 

    # model_path is the path where your trained Random Forest model is saved
    model_path = 'UNSW_NB15/attempt_3/models/random_forest_model.joblib'

    # Call the evaluate_model function
    evaluate_model('UNSW_NB15/transformation/transformed_test_file.csv', model_path, required_features, categorical_features, columns_to_scale)







# def transform(file_path: str) -> pd.DataFrame:
#     """Main function to transform the dataset for evaluation."""
#     df = load_dataset(file_path)
#     print(f"Loaded dataset with shape: {df.shape}")


#     # Define required columns (as used during training)
#     REQUIRED_FEATURES = [
#         'dur',
#         'service',
#         'state',
#         'spkts',
#         'dpkts',
#         'sbytes',
#         'dbytes',
#         'rate',
#         'sload',
#         'dload',
#         'sloss',
#         'dloss',
#         'sinpkt',
#         'dinpkt',
#         'sjit',
#         'djit',
#         'stcpb',
#         'dtcpb',
#         'tcprtt',
#         'synack',
#         'ackdat',
#         'smean',
#         'dmean',
#         'response_body_len',
#         'ct_src_dport_ltm',
#         'ct_dst_sport_ltm',
#         'ct_flw_http_mthd',
#         'label',
#     ]
#     df = align_features(df, REQUIRED_FEATURES)
#     print(f"Aligned features. New shape: {df.shape}")





#     df = handle_missing_values(df)
#     print("Handled missing values.")

#     # Final validation
#     assert set(df.columns) == set(REQUIRED_FEATURES), "Transformed data does not match required features!"
#     print("Validation successful: Features match required features.")

#     return df
