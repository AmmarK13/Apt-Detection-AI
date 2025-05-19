from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load the cleaned training dataset
df = pd.read_csv("UNSW_NB15/cleaned_data/training_outliers_final.csv")
print("dataset loaded")


# Step 1: Remove low-variance features
var_thresh = VarianceThreshold(threshold=0.01)  # Adjust threshold if needed
df_var = df.drop(columns=['label', 'attack_cat'])  # Exclude target labels
df_var = pd.DataFrame(var_thresh.fit_transform(df_var), columns=df_var.columns[var_thresh.get_support()])
print("step 1 done")

# Step 2: Compute Mutual Information (MI)
X = df_var
y = df['label']  # Using 'label' as the classification target
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_scores_df = pd.DataFrame({'Feature': X.columns, 'MI Score': mi_scores})
mi_scores_df = mi_scores_df.sort_values(by='MI Score', ascending=False)

# Keep features with MI > threshold (e.g., 0.01)
selected_features_mi = mi_scores_df[mi_scores_df['MI Score'] > 0.01]['Feature'].tolist()
df_selected = df[selected_features_mi + ['label', 'attack_cat']]  # Keep target variables

print("step 2 done")



# Step 3: Recursive Feature Elimination (RFE) with Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(rf, n_features_to_select=10)  # Select top 10 features
X_rfe = df_selected.drop(columns=['label', 'attack_cat'])
rfe.fit(X_rfe, y)

# Keep only selected features
selected_features_rfe = X_rfe.columns[rfe.support_].tolist()
df_final = df[selected_features_rfe + ['label', 'attack_cat']]  # Final dataset
print("step 3 done")


print(f"Selected Features ({len(selected_features_rfe)}):", selected_features_rfe)


df_final.to_csv("UNSW_NB15/cleaned_data/training_selected_features.csv", index=False)
print("Feature selection completed and saved dataset!")
