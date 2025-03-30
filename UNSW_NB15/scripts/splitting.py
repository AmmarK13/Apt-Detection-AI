import pandas as pd
from sklearn.model_selection import train_test_split

# Load the cleaned dataset
df = pd.read_csv("UNSW_NB15/cleaned_data/training_selected_features.csv")  

# Define features and target
X = df.drop(columns=["label", "attack_cat"])  # Features
y = df["label"]  # Target variable

# Split data: 70% training, 30% temporary (to be split into validation & test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

# Split temporary set into 20% validation & 10% test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42, stratify=y_temp)

# Save the splits
X_train.to_csv("UNSW_NB15/prepared_data/X_train.csv", index=False)
y_train.to_csv("UNSW_NB15/prepared_data/y_train.csv", index=False)
X_val.to_csv("UNSW_NB15/prepared_data/X_val.csv", index=False)
y_val.to_csv("UNSW_NB15/prepared_data/y_val.csv", index=False)
X_test.to_csv("UNSW_NB15/prepared_data/X_test.csv", index=False)
y_test.to_csv("UNSW_NB15/prepared_data/y_test.csv", index=False)

print("Data successfully split and saved!")
