import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Load preprocessed dataset
file_path = "/home/kay/Documents/Workspace-S25/SE/SeProject/APT DETECTION/z.after mids progress/dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX_CLEANED_BALANCED_CLEAN.csv"
df = pd.read_csv(file_path)

# ==================================================================
# 1. PROTECTED FEATURES CONFIGURATION (REVISED)
# ==================================================================
MANDATORY_FEATURES = {
    'SYN Flag Count': 'SYN Flood Detection',
    'Flow Duration': 'Attack Duration Analysis',
    'Fwd Packet Length Max': 'Payload Size Detection',
    'Flow Packets/s': 'Traffic Volume Spike',
    'URG Flag Count': 'Urgent Flag Patterns',
    'Bwd Packets/s': 'Reflection Attack Indicator'
}  # Removed 'Packet Length Variance' as it's not in original data

PROTECTED_FEATURES = list(MANDATORY_FEATURES.keys()) + ['Label']

# ==================================================================
# 2. UPDATED FEATURE SELECTION PIPELINE (WITH SAFETY CHECKS)
# ==================================================================
def feature_selection_pipeline(data):
    # Initial validation
    missing_mandatory = [f for f in MANDATORY_FEATURES if f not in data.columns]
    if missing_mandatory:
        raise ValueError(f"Critical features missing: {missing_mandatory}")
    
    # [Keep previous Phase 1 & 2 code...]
    
    # ========== PHASE 3 REVISION ==========
    # Modified mutual information section
    y = data["Label"].copy()
    X = data.drop(columns=["Label"], errors="ignore")
    
    if not X.empty:
        mi_scores = mutual_info_classif(X, y, random_state=42)
        mi_df = pd.DataFrame({"Feature": X.columns, "MI_Score": mi_scores})
        mi_threshold = np.quantile(mi_scores, 0.25)
        
        # Get existing features only
        top_features = mi_df[mi_df["MI_Score"] > mi_threshold]["Feature"].tolist()
        existing_mandatory = [f for f in MANDATORY_FEATURES if f in data.columns]
        top_features = list(set(top_features + existing_mandatory))
        
        # Filter to existing columns
        valid_features = [f for f in top_features if f in data.columns]
        data = data[valid_features + ["Label"]]
    
    # ========== PHASE 4 REVISION ==========
    if len(data.columns) > 2:
        X = data.drop(columns=["Label"])
        y = data["Label"]
        
        model = XGBClassifier(
            n_estimators=100,
            random_state=42,
            tree_method='hist',
            scale_pos_weight=2.0  # Explicit class balance
        )
        model.fit(X, y)
        
        importance = model.feature_importances_
        feat_imp = pd.DataFrame({"Feature": X.columns, "Importance": importance})
        
        # Dynamic threshold with fallback
        try:
            imp_threshold = np.quantile(importance, 0.2)
        except:
            imp_threshold = 0.001
            
        final_features = feat_imp[feat_imp["Importance"] > imp_threshold]["Feature"].tolist()
        final_features += [f for f in MANDATORY_FEATURES if f in data.columns]
        final_features = list(set(final_features))  # Remove duplicates
        
        data = data[final_features + ["Label"]]
    
    return data

# ==================================================================
# 3. EXECUTION & VALIDATION
# ==================================================================
print("Starting feature selection pipeline...")
df_final = feature_selection_pipeline(df.copy())

# Final validation
print("\n=== FINAL VALIDATION ===")
print("Mandatory features present:", [f for f in MANDATORY_FEATURES if f in df_final.columns])
print("Dataset shape:", df_final.shape)
print("Class balance:\n", df_final["Label"].value_counts())
print("NaN values:", df_final.isna().sum().sum())

# Save results
output_path = file_path.replace("BALANCED_CLEAN.csv", "OPTIMAL_FEATURES.csv")
df_final.to_csv(output_path, index=False)
print(f"\nDataset saved to: {output_path}")

# Visualization
plt.figure(figsize=(12,6))
pd.Series(df_final.drop(columns=["Label"]).columns).value_counts().plot.pie()
plt.title("Final Feature Composition")
plt.show()