import pandas as pd
import numpy as np
import xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class FeatureSelector:
    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss',
            enable_categorical=True  # Enable categorical feature support
        )
        self.feature_importance = None
        
    def fit(self, X, y):
        """Train the model and calculate feature importance"""
        # Split data for early stopping
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Fit without early stopping parameter
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Get feature importance (using gain for better interpretation)
        importance_dict = self.model.get_booster().get_score(importance_type='gain')
        
        # Convert to DataFrame with all features (including those with zero importance)
        importance_df = pd.DataFrame(
            [(col, importance_dict.get(col, 0)) for col in X.columns],
            columns=['feature', 'importance']
        )
        
        self.feature_importance = importance_df.sort_values('importance', ascending=False)
        
    def plot_importance(self, top_n=20):
        """Plot feature importance"""
        plt.figure(figsize=(12, 8))
        sns.barplot(
            data=self.feature_importance.head(top_n),
            x='importance',
            y='feature'
        )
        plt.title(f'Top {top_n} Most Important Features')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig('d:/University/Software Engineering/Project/Apt-Detection-AI/webAttacks/results/feature_importance.png')
        plt.close()
        
    def select_features(self, importance_threshold=0.01):
        """Select features based on importance threshold"""
        selected_features = self.feature_importance[
            self.feature_importance['importance'] >= importance_threshold
        ]['feature'].tolist()
        return selected_features

def main():
    import os
    from sklearn.preprocessing import LabelEncoder
    
    # Create results directory if it doesn't exist
    results_dir = "d:/University/Software Engineering/Project/Apt-Detection-AI/webAttacks/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load the dataset with absolute path
    df = pd.read_csv("d:/University/Software Engineering/Project/Apt-Detection-AI/webAttacks/data/removed_features.csv")
    
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Use label encoding instead of one-hot encoding to save memory
    df_processed = df.copy()
    if categorical_cols:
        print(f"Label encoding {len(categorical_cols)} categorical columns...")
        for col in categorical_cols:
            if col != 'classification':  # Don't encode the target if it's categorical
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    
    # Separate features and target
    X = df_processed.drop('classification', axis=1)
    y = df_processed['classification']
    
    print(f"Dataset shape after encoding: {X.shape}")
    
    # Initialize feature selector
    selector = FeatureSelector()
    
    # Fit the model
    selector.fit(X, y)
    
    # Select important features
    selected_features = selector.select_features(importance_threshold=0.01)
    
    # Create dataset with selected features
    df_selected = df[selected_features + ['classification']]
    
    # Plot feature importance with absolute path
    selector.plot_importance(top_n=20)
    
    # Save results
    df_selected.to_csv("d:/University/Software Engineering/Project/Apt-Detection-AI/webAttacks/data/selected_features.csv", index=False)
    
    # Save feature importance scores
    selector.feature_importance.to_csv(
        "d:/University/Software Engineering/Project/Apt-Detection-AI/webAttacks/results/feature_importance_scores.csv",
        index=False
    )
    
    # Print summary
    print("\nFeature Selection Summary:")
    print(f"Original number of features: {X.shape[1]}")
    print(f"Number of selected features: {len(selected_features)}")
    print("\nTop 10 most important features:")
    for idx, row in selector.feature_importance.head(10).iterrows():
        print(f"- {row['feature']}: {row['importance']:.4f}")
    
    # Save feature importance scores
    selector.feature_importance.to_csv(
        "data/feature_importance_scores.csv",
        index=False
    )

if __name__ == "__main__":
    main()