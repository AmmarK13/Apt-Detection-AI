import pandas as pd

def remove_recommended_features():
    # Read the original dataset
    df = pd.read_csv("d:/University/Software Engineering/Project/Apt-Detection-AI/webAttacks/data/robust_scaled_csic_database.csv")
    
    # Read recommendations
    recommendations = pd.read_csv("d:/University/Software Engineering/Project/Apt-Detection-AI/webAttacks/data/feature_removal_recommendations.csv")
    
    # Get features marked for removal
    features_to_remove = recommendations[recommendations['status'].isin(['Remove', 'Consider Removing'])]['feature'].tolist()
    
    # Print information about removal
    print(f"Total features before removal: {len(df.columns)}")
    print("\nFeatures being removed:")
    for feature in features_to_remove:
        print(f"- {feature}")
    
    # Remove the features
    df_cleaned = df.drop(columns=features_to_remove)
    
    # Save the cleaned dataset
    output_path = "d:/University/Software Engineering/Project/Apt-Detection-AI/webAttacks/data/removed_features_second.csv"
    df_cleaned.to_csv(output_path, index=False)
    
    print(f"\nTotal features after removal: {len(df_cleaned.columns)}")
    for feature in df_cleaned.columns:
        print(f"- {feature}")
    print(f"\nCleaned dataset saved to: {output_path}")

if __name__ == "__main__":
    remove_recommended_features()