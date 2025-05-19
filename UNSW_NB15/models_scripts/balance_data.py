# from imblearn.over_sampling import SMOTE
# import pandas as pd

# # Load the training dataset
# df_train = pd.read_csv("UNSW_NB15/cleaned_data/training_selected_features.csv")  # Update with correct path
# print("Training data loaded.")

# # Separate features and target
# X_train = df_train.drop(columns=['label', 'attack_cat'])  # Drop target labels
# y_train = df_train['label']

# # Apply SMOTE
# smote = SMOTE(sampling_strategy='auto', random_state=42)  # 'auto' balances both classes equally
# X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# # Convert back to DataFrame
# df_train_resampled = pd.DataFrame(X_train_resampled, columns=X_train.columns)
# df_train_resampled['label'] = y_train_resampled  # Add back the label column

# # Save the resampled dataset
# df_train_resampled.to_csv("UNSW_NB15/cleaned_data/training_balanced_smote.csv", index=False)
# print("SMOTE applied. Balanced training data saved.")
