import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from matplotlib.backends.backend_pdf import PdfPages

# Load the encoded dataset
df_encoded = pd.read_csv("data/Cleaned_Preprocessed_Numerical.csv")

# Define the target column
target_col = 'classification'

# -------------------------------
# Numerical Analysis on Encoded Data
# -------------------------------

# Create a PDF file to save plots
with PdfPages("output_plots.pdf") as pdf:
    # 1. Correlation Matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix (Encoded Data)")
    pdf.savefig()  # Save the current figure to the PDF
    plt.close()  # Close the figure to free memory

    # Calculate correlation with target for each feature
    correlations = df_encoded.corr()[target_col].sort_values(ascending=False)
    print("\nCorrelations with Target (classification) after encoding:")
    print(correlations)

    # Flag potential leakage features (using a threshold, e.g., 0.8)
    leakage_threshold = 0.8
    leakage_features = correlations[abs(correlations) > leakage_threshold].index.tolist()
    if target_col in leakage_features:
        leakage_features.remove(target_col)
    if leakage_features:
        print("Warning: Potential leakage features based on high correlation:", leakage_features)
    else:
        print("No potential leakage features detected based on correlation threshold.")

    # 2. Mutual Information for Encoded Features
    X_encoded = df_encoded.drop(target_col, axis=1)
    y = df_encoded[target_col]
    mi_scores = mutual_info_classif(X_encoded, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=X_encoded.columns).sort_values(ascending=False)
    print("\nTop 10 Features by Mutual Information (Encoded Data):")
    print(mi_series.head(10))

    # Optionally, flag features with high mutual information (e.g., > 0.1)
    mi_threshold = 0.1
    leakage_mi_features = mi_series[mi_series > mi_threshold].index.tolist()
    if leakage_mi_features:
        print("Warning: The following encoded features have high mutual information with the target:",
              leakage_mi_features)
    else:
        print("No categorical features show signs of leakage based on mutual information.")

print("\nEDA complete on encoded data.")
