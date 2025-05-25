import pandas as pd
from sklearn.model_selection import train_test_split

# Load your balanced dataset
df = pd.read_csv('dataset_balanced.csv')

# Separate features and label
X = df.drop('Label', axis=1)
y = df['Label']

# Split 80% train_val, 20% test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Recombine features and labels for saving
train_val_df = X_train_val.copy()
train_val_df['Label'] = y_train_val

test_df = X_test.copy()
test_df['Label'] = y_test

# Save to CSVs
train_val_df.to_csv('train_val.csv', index=False)
test_df.to_csv('test.csv', index=False)

print("✅ Split complete!")
print(f"📝 Training+Validation set: {train_val_df.shape[0]} rows")
print(f"🧪 Testing set: {test_df.shape[0]} rows")
