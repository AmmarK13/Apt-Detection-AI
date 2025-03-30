# check imbalance in traing and testing models
import pandas as pd

# Load the training and testing datasets
train_data = pd.read_csv("UNSW_NB15/cleaned_data/training_selected_features.csv") 
test_data = pd.read_csv("UNSW_NB15/cleaned_data/testing_selected_features.csv")  

# Count class distribution
train_class_counts = train_data['label'].value_counts()
test_class_counts = test_data['label'].value_counts()

# Print results
print("Training Data Class Distribution:")
print(train_class_counts)
print("\nTesting Data Class Distribution:")
print(test_class_counts)
