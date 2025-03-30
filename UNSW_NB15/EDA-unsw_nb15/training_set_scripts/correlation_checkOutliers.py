import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("UNSW_NB15/cleaned_data/training_encoded.csv")


# Compute correlation matrix
corr_matrix = df.corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, cmap="coolwarm", annot=False, fmt=".2f")
plt.title("Feature Correlation Heatmap After Encoding")
plt.show()

# Identify highly correlated features (threshold = 0.9)
high_corr_pairs = []
threshold = 0.9
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

# Print highly correlated features
if high_corr_pairs:
    print("\n⚠️ Highly Correlated Features (>|0.9|):")
    for f1, f2, corr in high_corr_pairs:
        print(f"{f1} and {f2} - Correlation: {corr:.2f}")
else:
    print("\n✅ No highly correlated features found.")



# Check number of features before
num_features_before = df.shape[1]

# Compute correlation matrix
threshold = 0.9
corr_matrix = df.corr()

# Get upper triangle of the correlation matrix
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find columns to drop
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
print(f"Features to be dropped: {to_drop}")

# Drop the highly correlated features
df_cleaned = df.drop(columns=to_drop)

# Check number of features after
num_features_after = df_cleaned.shape[1]

print(f"Number of features before dropping: {num_features_before}")
print(f"Number of features after dropping: {num_features_after}")



# checks--------------------------
print(df_cleaned.duplicated().sum())  # Should be 0 ideally
print(df_cleaned.isna().sum().sum())  # Should print 0 if no missing values
print(df_cleaned.shape)
print(df_cleaned.columns.tolist())  # Should not contain the dropped features
plt.figure(figsize=(12, 8))
sns.heatmap(df_cleaned.corr(), cmap="coolwarm", annot=False, fmt=".2f")
plt.title("Feature Correlation Heatmap After Dropping Highly Correlated Features")
plt.show()


# still high coeerr?
print("till high correlations?")
corr_matrix = df_cleaned.corr()
high_corr_pairs = corr_matrix[(corr_matrix > 0.85) | (corr_matrix < -0.85)]
print(high_corr_pairs.dropna(how='all').dropna(axis=1, how='all'))



# LABEL LEAKAGE =========================================================================

print("\n\nLABEL LEAKAGE")
# Compute correlation matrix
correlation_matrix = df_cleaned.corr()

# Check correlation of 'label' and 'attack_cat' with all other features
label_corr = correlation_matrix["label"].abs().sort_values(ascending=False)
attack_cat_corr = correlation_matrix["attack_cat"].abs().sort_values(ascending=False)

# Print features highly correlated with 'label' or 'attack_cat'
print("Highly Correlated Features with 'label' (>0.85):")
print(label_corr[label_corr > 0.85])

print("\nHighly Correlated Features with 'attack_cat' (>0.85):")
print(attack_cat_corr[attack_cat_corr > 0.85])


print("\n\nINDIRECT LEAKAGE\nbit more caucious check with 0.75")

# Set correlation threshold for indirect leakage
leakage_threshold = 0.75

# Compute correlation matrix
correlation_matrix = df.corr()

# Check for indirect leakage with 'label'
high_corr_label = correlation_matrix['label'][abs(correlation_matrix['label']) > leakage_threshold]
print("Indirect Leakage Check: Features highly correlated with 'label' (>0.75):")
print(high_corr_label)

# Check for indirect leakage with 'attack_cat'
high_corr_attack_cat = correlation_matrix['attack_cat'][abs(correlation_matrix['attack_cat']) > leakage_threshold]
print("\nIndirect Leakage Check: Features highly correlated with 'attack_cat' (>0.75):")
print(high_corr_attack_cat)





df_cleaned.to_csv("UNSW_NB15/cleaned_data/training_outliers_final.csv", index=False)
print("Dataset saved successfully!")
