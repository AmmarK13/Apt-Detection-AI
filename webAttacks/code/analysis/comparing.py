import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Load your dataset (replace 'your_dataset.csv' with your file path)
df = pd.read_csv('D:/University/Software Engineering/Project/data/csic_reduced_minmaxScaled.csv')  # or pd.read_excel() for Excel files

# Check if loaded correctly
print("First 5 rows:\n", df.head())
print("\nDataset shape:", df.shape)

# Prepare features/target
X = df.drop("classification", axis=1)
y = df["classification"]

# Logistic Regression
lr = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
lr_scores = cross_val_score(lr, X, y, cv=5, scoring='roc_auc')
print(f"\nLogistic Regression AUC: {np.mean(lr_scores):.3f}")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=5)
rf_scores = cross_val_score(rf, X, y, cv=5, scoring='roc_auc')
print(f"Random Forest AUC: {np.mean(rf_scores):.3f}")