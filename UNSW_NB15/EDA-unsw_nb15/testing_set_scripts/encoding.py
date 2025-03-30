import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from category_encoders import TargetEncoder
import category_encoders as ce

# Load the cleaned training dataset
df_cleaned = pd.read_csv("UNSW_NB15/cleaned_data/testing_outliers_v2.csv")


print(df_cleaned.columns)  # List all column names
print(df_cleaned.shape)  # Compare with original

# 🚀 Step 1: Convert 'Label' to int
if "label" in df_cleaned.columns:
    df_cleaned["label"] = df_cleaned["label"].astype(int)
    print("label converted to int")
else:
    print("error 1")


# 🚀 Step 2: Label Encoding for 'attack_cat'
if "attack_cat" in df_cleaned.columns:
    label_encoder = LabelEncoder()
    df_cleaned["attack_cat"] = label_encoder.fit_transform(df_cleaned["attack_cat"])
    print("Label Encoding for: attack_cat")
else:
    print("error 2")

# 🚀 Step 3: target Encoding for 'service', 'state', and 'proto'

# Define categorical columns for target encoding
target_encoding_cols = ["proto", "service", "state"]

# Initialize Target Encoder
target_encoder = ce.TargetEncoder(cols=target_encoding_cols)

# Apply Target Encoding (fit on training data, transform both train & test)
df_cleaned[target_encoding_cols] = target_encoder.fit_transform(df_cleaned[target_encoding_cols], df_cleaned["label"])
print("Target Encoding done on proto,service,")




print("\n\nsanity checks\n")

print("duplicates:" )
print(df_cleaned.duplicated().sum())
df_cleaned = df_cleaned.drop_duplicates()
print("dupps removed")

print(df_cleaned.shape)  # Compare with original

print("types:")
print(df_cleaned.columns)


# ✅ Save the encoded dataset
df_cleaned.to_csv("UNSW_NB15/cleaned_data/testing_encoded.csv", index=False)