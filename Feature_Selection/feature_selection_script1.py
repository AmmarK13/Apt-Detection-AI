from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load your cleaned dataset
file_path=r"D:\4th semester\SE\project\final_final\2-21-2018_final_final.csv"
df = pd.read_csv(file_path)

# Separate features and target variable
X = df.drop(columns=["Label"])  # Assuming "Label" is the target column
y = df["Label"]

# Train a basic Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Get feature importance scores
feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
feature_importances = feature_importances.sort_values(ascending=False)

# Print top features
print("Top Important Features:\n", feature_importances.head(10))

# Plot feature importances
import matplotlib.pyplot as plt
feature_importances.head(10).plot(kind="barh", figsize=(8,6), title="Top 10 Important Features")
plt.show()
