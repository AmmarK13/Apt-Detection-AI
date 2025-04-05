import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataSplitter:
    def __init__(self, input_file: str, output_dir: str, test_size: float = 0.2, random_state: int = 42):
        """Initialize the DataSplitter with input and output paths.

        Args:
            input_file (str): Path to the engineered features CSV file
            output_dir (str): Directory to save train/test splits
            test_size (float): Proportion of dataset to include in the test split
            random_state (int): Random state for reproducibility
        """
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.test_size = test_size
        self.random_state = random_state
        self.df = None
        self.scaler = StandardScaler()

    def load_data(self):
        """Load the engineered features dataset."""
        try:
            self.df = pd.read_csv(self.input_file)
            logger.info(f"Loaded dataset with shape: {self.df.shape}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def preprocess_features(self):
        """Preprocess features by separating numerical and categorical columns."""
        try:
            # Identify binary columns (HTTP methods and attack patterns)
            binary_columns = [col for col in self.df.columns if col.startswith(('method_', 'attack_'))]
            
            # Identify numerical columns (excluding classification and binary features)
            numerical_columns = self.df.select_dtypes(include=['int64', 'float64']).columns
            numerical_columns = [col for col in numerical_columns 
                               if col not in binary_columns and col != 'classification']

            logger.info(f"Identified {len(binary_columns)} binary features and {len(numerical_columns)} numerical features")
            return numerical_columns, binary_columns

        except Exception as e:
            logger.error(f"Error in feature preprocessing: {e}")
            raise

    def split_data(self):
        """Split the dataset into training and testing sets with proper scaling."""
        try:
            # Preprocess features
            numerical_columns, binary_columns = self.preprocess_features()
            
            # Separate features and target
            y = self.df['classification']
            
            # Split numerical features
            X_numerical = self.df[numerical_columns]
            
            # Split binary features
            X_binary = self.df[binary_columns]
            
            # Scale numerical features
            X_numerical_scaled = self.scaler.fit_transform(X_numerical)
            X_numerical_scaled = pd.DataFrame(X_numerical_scaled, columns=numerical_columns)
            
            # Combine scaled numerical and binary features
            X = pd.concat([X_numerical_scaled, X_binary], axis=1)

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y  # Ensure balanced split
            )

            # Recombine features and target for saving
            train_df = pd.concat([X_train, y_train], axis=1)
            test_df = pd.concat([X_test, y_test], axis=1)

            logger.info(f"Training set shape: {train_df.shape}")
            logger.info(f"Testing set shape: {test_df.shape}")

            return train_df, test_df

        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            raise

    def save_splits(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Save the training and testing splits to CSV files."""
        try:
            # Create output directory if it doesn't exist
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Save splits
            train_path = self.output_dir / 'train_data.csv'
            test_path = self.output_dir / 'test_data.csv'

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            logger.info(f"Training data saved to: {train_path}")
            logger.info(f"Testing data saved to: {test_path}")

        except Exception as e:
            logger.error(f"Error saving splits: {e}")
            raise

    def process(self):
        """Execute the complete data splitting pipeline."""
        logger.info("Starting data splitting process...")
        
        self.load_data()
        train_df, test_df = self.split_data()
        self.save_splits(train_df, test_df)
        
        logger.info("Data splitting completed successfully")

def main():
    # Define input and output paths
    input_file = "data/features/engineered_features.csv"
    output_dir = "data/split"
    
    # Initialize and run the splitter
    splitter = DataSplitter(input_file, output_dir)
    try:
        splitter.process()
    except Exception as e:
        logger.error(f"Error during data splitting: {e}")

if __name__ == "__main__":
    main()