import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.model_selection import train_test_split

def balance_dataset(file_path, target_column, method='oversample', output_path='balanced_dataset.csv'):
    """
    Balances the dataset using oversampling (SMOTE) or undersampling (RandomUnderSampler) and saves the result.
    
    Parameters:
        file_path (str): Path to the dataset file (CSV).
        target_column (str): The column to balance (usually the target variable).
        method (str): Balancing method: 'oversample' (SMOTE) or 'undersample' (RandomUnderSampler).
        output_path (str): The path where the balanced dataset will be saved.
    """
    try:
        # Load dataset
        data = pd.read_csv(file_path)
        
        # Prepare features and target
        X = data.drop(target_column, axis=1)  # Features
        y = data[target_column]               # Target variable
        
        # Split dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Check initial class distribution
        print(f"Initial Class Distribution in '{target_column}':")
        print(Counter(y_train))
        print("\n")
        
        # Balance the dataset using the chosen method
        if method == 'oversample':
            # Apply SMOTE (oversampling)
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            print(f"Class Distribution after Oversampling (SMOTE):")
        elif method == 'undersample':
            # Apply RandomUnderSampler (undersampling)
            rus = RandomUnderSampler(random_state=42)
            X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)
            print(f"Class Distribution after Undersampling:")
        else:
            raise ValueError("Method should be 'oversample' or 'undersample'.")
        
        # Check class distribution after balancing
        print(Counter(y_train_resampled))
        
        # Combine the resampled features and target into a DataFrame
        X_train_resampled = pd.DataFrame(X_train_resampled, columns=X.columns)
        y_train_resampled = pd.DataFrame(y_train_resampled, columns=[target_column])
        balanced_data = pd.concat([X_train_resampled, y_train_resampled], axis=1)
        
        # Save the balanced dataset to the specified output path
        balanced_data.to_csv(output_path, index=False)
        print(f"\nBalanced dataset saved to {output_path}")
        
        return X_train_resampled, X_test, y_train_resampled, y_test
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Usage Example
# Replace 'your_dataset.csv' with the path to your dataset and 'classification' with your target column name
file_path = 'D:/University/Software Engineering/Project/data/csic_reduced_minmaxScaled.csv'
target_column = 'classification'  # Adjust this to your target column
output_path = 'D:/University/Software Engineering/Project/data/balanced_dataset.csv'  # Path where the balanced dataset will be saved

X_train_balanced, X_test, y_train_balanced, y_test = balance_dataset(file_path, target_column, method='oversample', output_path=output_path)
