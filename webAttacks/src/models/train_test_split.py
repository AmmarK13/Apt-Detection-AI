import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split

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

    def load_data(self):
        """Load the engineered features dataset."""
        try:
            self.df = pd.read_csv(self.input_file)
            logger.info(f"Loaded dataset with shape: {self.df.shape}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def split_data(self):
        """Split the dataset into training and testing sets."""
        try:
            # Separate features and target
            X = self.df.drop('classification', axis=1)
            y = self.df['classification']

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