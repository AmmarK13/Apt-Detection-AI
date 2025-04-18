import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the data
df = pd.read_csv("UNSW_NB15/cleaned_data/reduced_training_set.csv")

# Count the number of rows where 'service' column contains '-'
missing_service_count = df[df['service'] == '-'].shape[0]
total_rows = df.shape[0]
print(f"Total number of rows: {total_rows}")
print(f"Number of rows with '-' in 'service' column: {missing_service_count}")

# Impute missing values in 'service' column with the most frequent value
imputer = SimpleImputer(strategy="most_frequent")
df['service'] = imputer.fit_transform(df[['service']]).ravel()  # Use .ravel() to flatten the result
print("Missing values in 'service' column have been imputed.")

# ================ ENCODING ==================================================
# Define your categorical and text/meta features
categorical_features = ['label', 'is_sm_ips_ports', 'service', 'state', 'attack_cat', 'proto']

# Preview unique values for decision-making
for col in categorical_features:
    print(f"{col} unique values:", df[col].nunique(), df[col].unique()[:10])

# Apply Label Encoding to 'label' and 'is_sm_ips_ports' separately
label_encoder = LabelEncoder()

# Apply label encoding to the 'label' and 'is_sm_ips_ports' columns
df['label'] = label_encoder.fit_transform(df['label'])
df['is_sm_ips_ports'] = label_encoder.fit_transform(df['is_sm_ips_ports'])

# Create transformers for the remaining categorical features
onehot_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),  # handle missing values
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Apply OneHotEncoding to the other categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ("onehot", onehot_transformer, ['service', 'state', 'attack_cat', 'proto'])  # Apply OneHotEncoder to other categorical features
    ],
    remainder='passthrough'  # leave other columns untouched
)

# Fit and transform the data
# Step 2: Label Encoding for categorical columns
label_encoder = LabelEncoder()

# Apply label encoding to 'service', 'state', 'attack_cat', and 'proto'
df['service'] = label_encoder.fit_transform(df['service'])
df['state'] = label_encoder.fit_transform(df['state'])
df['attack_cat'] = label_encoder.fit_transform(df['attack_cat'])
df['proto'] = label_encoder.fit_transform(df['proto'])

# Step 3: Display the unique values of each column to confirm label encoding
print(f"label unique values: {df['service'].unique()}")
print(f"state unique values: {df['state'].unique()}")
print(f"attack_cat unique values: {df['attack_cat'].unique()}")
print(f"proto unique values: {df['proto'].unique()}")

# Step 4: Optionally, save the cleaned and encoded DataFrame to a CSV file
df.to_csv('UNSW_NB15/cleaned_attempt2/1_encoded_data.csv', index=False)
print("Encoded data has been saved to 'encoded_data.csv'")



# X_encoded = preprocessor.fit_transform(df)
# print("Encoding complete. Shape of the encoded data:", X_encoded.shape)

# # Extract feature names from the one-hot encoding step
# onehot_feature_names = preprocessor.transformers_[0][1].named_steps["onehot"].get_feature_names_out()

# # Combine the column names for label encoding with one-hot encoded columns
# columns = list(label_encoder.classes_) + list(onehot_feature_names)


# # Print the number of columns in the transformed data
# print("Shape of the transformed data:", X_encoded.shape)

# # Check the number of one-hot encoded columns
# onehot_feature_names = preprocessor.transformers_[0][1].named_steps["onehot"].get_feature_names_out()

# # Print the number of one-hot encoded features
# print("Number of one-hot encoded features:", len(onehot_feature_names))

# # If the number of columns doesn't match, manually check the number of unique values in categorical columns
# # You can assign the correct number of feature names based on the unique categories in your columns

# print("Shape of the transformed data:", X_encoded.shape)
# print("Expected number of columns:", len(columns))  # You can check the number of expected columns in 'columns'

# onehot_feature_names = preprocessor.transformers_[0][1].named_steps["onehot"].get_feature_names_out()
# print("Number of one-hot encoded features:", len(onehot_feature_names))

# # Check the columns that were one-hot encoded
# print("Original categorical columns to be encoded:", ['service', 'state', 'attack_cat', 'proto'])

# # Check the feature names of the one-hot encoder
# print("One-hot encoded feature names:", onehot_feature_names)






# # Assign column names correctly
# columns = list(label_encoder.classes_) + list(onehot_feature_names)

# # Create a DataFrame with the encoded data
# encoded_df = pd.DataFrame(X_encoded, columns=columns)

# # Save the DataFrame to a CSV file
# encoded_df.to_csv('UNSW_NB15/cleaned_attempt2/1_encoded_data.csv', index=False)

# print("Encoded data has been saved to 'encoded_data.csv'")
