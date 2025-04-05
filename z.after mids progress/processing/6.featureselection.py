import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Load preprocessed dataset
file_path = "/home/kay/Documents/Workspace-S25/SE/SeProject/APT DETECTION/z.after mids progress/dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX_CLEANED_BALANCED_CLEAN.csv"
df = pd.read_csv(file_path)

# ------------------------------
# 1. Remove Low-Variance Features (with protection)
# ------------------------------
PROTECTED_FEATURES = ['SYN Flag Count', 'URG Flag Count', 'Flow Duration', 'Label']

# Identify protected and non-protected features
non_protected = [col for col in df.columns if col not in PROTECTED_FEATURES]

# Apply variance threshold only to non-protected features
thresh = 0.05 * (1 - 0.05)
selector = VarianceThreshold(threshold=thresh)
df_non_protected = df[non_protected].select_dtypes(include="number")
selector.fit(df_non_protected)
low_var_mask = selector.get_support()
selected_non_protected = df_non_protected.columns[low_var_mask]

# Combine protected features with selected non-protected
selected_features = list(selected_non_protected) + [f for f in PROTECTED_FEATURES if f in df.columns]
df = df[selected_features]
print(f"Removed {len(non_protected) - len(selected_non_protected)} low-variance features")

# ------------------------------
# 2. Enhanced Correlation Analysis & Removal
# ------------------------------
# Check remaining features
numeric_cols = df.select_dtypes(include="number").columns.drop("Label", errors="ignore")
if len(numeric_cols) == 0:
    raise ValueError("No numeric features remaining after variance threshold!")

# Calculate correlations
corr_matrix = df[numeric_cols].corr()

# Visualization
plt.figure(figsize=(20, 15))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', mask=mask)
plt.title("Feature Correlation Matrix", fontsize=14)
plt.show()

# Correlation removal with protection
corr_threshold = 0.95
upper = corr_matrix.where(mask)
to_drop = [col for col in upper.columns if (upper[col].abs() > corr_threshold).any()]

# Protect essential DDoS features
to_drop = [col for col in to_drop if col not in PROTECTED_FEATURES]

# Safe removal
existing_drops = [col for col in to_drop if col in df.columns]
df = df.drop(columns=existing_drops, errors="ignore")
print(f"Dropped {len(existing_drops)} highly correlated features: {existing_drops}")

# ------------------------------
# 3. Robust Mutual Information Processing
# ------------------------------
# Pre-check validation
if "Label" not in df.columns:
    raise KeyError("Critical error: 'Label' column missing!")

if df.drop("Label", axis=1).shape[1] == 0:
    raise ValueError("All features removed! Adjust selection thresholds.")

# Handle NaNs
nan_count = df.drop("Label", axis=1).isna().sum().sum()
if nan_count > 0:
    print(f"Imputing {nan_count} missing values...")
    df = df.fillna(df.median(numeric_only=True))

# Prepare data for MI
y = df["Label"].copy()
X = df.drop(columns=["Label"], errors="ignore")

# ------------------------------
# 4. Model-Based Selection with Validation
# ------------------------------
# Mutual Information
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_df = pd.DataFrame({"Feature": X.columns, "MI_Score": mi_scores}).sort_values("MI_Score", ascending=False)

# Visualization
plt.figure(figsize=(12, 8))
sns.barplot(x="MI_Score", y="Feature", data=mi_df.head(20))
plt.title("Top 20 Features by Mutual Information Score")
plt.show()

# Select top features
top_features = mi_df[mi_df["MI_Score"] > 0.01]["Feature"].tolist()
print(f"Selected {len(top_features)} statistically important features")

# XGBoost feature importance
X_train, X_val, y_train, y_val = train_test_split(
    X[top_features], y, 
    test_size=0.2, 
    stratify=y, 
    random_state=42
)

model = XGBClassifier(
    n_estimators=100,
    random_state=42,
    tree_method='hist',
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# Get and sort importances
importance = model.feature_importances_
feat_imp = pd.DataFrame({
    "Feature": top_features,
    "Importance": importance
}).sort_values("Importance", ascending=False)

# Visualization
plt.figure(figsize=(12, 8))
sns.barplot(x="Importance", y="Feature", data=feat_imp.head(20))
plt.title("Top 20 Model-Based Important Features")
plt.show()

# Final selection
final_features = feat_imp[feat_imp["Importance"] > 0.005]["Feature"].tolist()
print(f"Final selected features ({len(final_features)}): {final_features}")

# ------------------------------
# 5. Final Validation & Saving
# ------------------------------
# Ensure final features exist
missing_features = [f for f in final_features if f not in df.columns]
if missing_features:
    print(f"Warning: {len(missing_features)} features missing in final set")
    final_features = [f for f in final_features if f in df.columns]

# Save final dataset
df_final = df[final_features + ["Label"]]
output_path = file_path.replace("BALANCED_CLEAN.csv", "FINAL_SELECTED.csv")
df_final.to_csv(output_path, index=False)
print(f"Final dataset saved to: {output_path}")
print(f"Final dataset shape: {df_final.shape}")

# Final validation check
print("\n=== Final Validation Checks ===")
print(f"Class distribution:\n{df_final['Label'].value_counts()}")
print(f"NaN values: {df_final.isna().sum().sum()}")
print(f"Feature types:\n{df_final.dtypes.value_counts()}")