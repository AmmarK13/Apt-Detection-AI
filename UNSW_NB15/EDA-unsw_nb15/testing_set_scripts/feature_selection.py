from sklearn.feature_selection import VarianceThreshold
import pandas as pd

# Load the cleaned testing dataset
df = pd.read_csv("UNSW_NB15/cleaned_data/testing_outliers_final.csv")
print("Dataset loaded")

# Define the features selected during training
selected_features_train = ['tcprtt', 'sbytes', 'dbytes', 'dinpkt', 'dmean', 'sinpkt', 'smean', 'dur', 'rate', 'dtcpb']

# Step 1: Remove low-variance features
var_thresh = VarianceThreshold(threshold=0.01)  # Adjust threshold if needed
df_var = df.drop(columns=['label', 'attack_cat'])  # Exclude target labels
df_var = pd.DataFrame(var_thresh.fit_transform(df_var), columns=df_var.columns[var_thresh.get_support()])
print("Step 1 done")

# Ensure only the **same** features are selected as in training
common_features = [feat for feat in selected_features_train if feat in df_var.columns]

# Check if there are any missing features
missing_features = set(selected_features_train) - set(common_features)
if missing_features:
    print(f"Warning: The following training features are missing in testing data: {missing_features}")

# Keep only selected features
df_final = df[common_features + ['label', 'attack_cat']]  # Keep labels
print(f"Final Selected Features ({len(common_features)}):", common_features)

# Save the final dataset
df_final.to_csv("UNSW_NB15/cleaned_data/testing_selected_features.csv", index=False)
print("Feature selection completed and saved dataset!")
