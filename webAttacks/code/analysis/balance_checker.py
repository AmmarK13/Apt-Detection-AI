import pandas as pd

def check_class_balance(file_path, target_column, threshold=0.4):
    """
    Checks the balance of a specific column in the dataset and returns yes/no answer.
    
    Parameters:
        file_path (str): Path to the dataset file (CSV).
        target_column (str): The column to check for balance (usually the target variable).
        threshold (float): Threshold for determining balance (default 0.4)
    """
    try:
        # Load dataset
        data = pd.read_csv(file_path)
        
        # Check balance of the target column
        class_distribution = data[target_column].value_counts(normalize=True)
        
        # Check if classes are balanced
        min_class_ratio = class_distribution.min()
        is_balanced = min_class_ratio >= threshold
        
        # Display results
        print(f"\nClass Distribution of '{target_column}':")
        print(class_distribution)
        print(f"\nIs the dataset balanced? {'Yes' if is_balanced else 'No'}")
        
        # Optionally, you can also check for visual representation
        try:
            import matplotlib.pyplot as plt
            class_distribution.plot(kind='bar', title=f"Class Distribution of {target_column}")
            plt.xlabel(target_column)
            plt.ylabel('Percentage')
            plt.show()
        except ImportError:
            print("Matplotlib is not installed. Install it to view the bar plot.")
        
        return "Yes" if is_balanced else "No"
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Error"

# Usage Example
file_path = 'D:/University/Software Engineering/Project/data/csic_reduced_minmaxScaled.csv'
target_column = 'classification'
check_class_balance(file_path, target_column)
