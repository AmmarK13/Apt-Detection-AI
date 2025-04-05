import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self, input_file: str, output_dir: str):
        """Initialize the DataCleaner with input and output paths.

        Args:
            input_file (str): Path to the input CSV file
            output_dir (str): Directory to save cleaned data
        """
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.df = None

    def load_data(self):
        """Load the dataset from CSV file."""
        try:
            self.df = pd.read_csv(self.input_file)
            logger.info(f"Loaded dataset with shape: {self.df.shape}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def remove_duplicates(self):
        """Remove duplicate entries from the dataset."""
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        logger.info(f"Removed {initial_rows - len(self.df)} duplicate rows")

    def handle_missing_values(self):
        """Handle missing values in the dataset."""
        # Fill missing values with appropriate strategies
        self.df = self.df.fillna('')
        logger.info("Handled missing values")

    def validate_data_types(self):
        """Validate and convert data types."""
        try:
            # Drop unnamed index column if it exists
            unnamed_cols = [col for col in self.df.columns if col.startswith('Unnamed:')]
            if unnamed_cols:
                self.df = self.df.drop(columns=unnamed_cols)
                logger.info(f"Dropped unnamed index columns: {unnamed_cols}")

            # Convert length to numeric if exists
            if 'length' in self.df.columns:
                self.df['length'] = pd.to_numeric(self.df['length'], errors='coerce').fillna(0).astype(int)

            # Ensure classification is integer
            self.df['classification'] = self.df['classification'].astype(int)
            logger.info("Data types validated and converted")
        except Exception as e:
            logger.error(f"Error in data type validation: {e}")
            raise

    def standardize_connection_values(self):
        """Standardize connection values in the dataset."""
        try:
            if 'connection' in self.df.columns:
                self.df['connection'] = self.df['connection'].replace('Connection: close', 'close')
                logger.info("Standardized connection values")
        except Exception as e:
            logger.error(f"Error in standardizing connection values: {e}")
            raise

    def extract_http_features(self):
        """Extract features from HTTP methods and attack types."""
        try:
            if 'method' in self.df.columns:
                # Create binary columns for each HTTP method
                methods = self.df['method'].unique()
                for method in methods:
                    self.df[f'method_{method}'] = (self.df['method'] == method).astype(int)
                logger.info(f"Created features for HTTP methods: {methods}")

            if 'url' in self.df.columns:
                # Extract attack patterns from URLs
                attack_patterns = {
                    'sql_injection': r'(?i)(union\s+select|insert\s+into|select\s+from|drop\s+table)',
                    'xss': r'(?i)(<script>|javascript:|onload=|onerror=)',
                    'path_traversal': r'(?i)(\.\./|\.\.\|\/etc\/|\/windows\/)',
                    'command_injection': r'(?i)(;\s*\w+\s*;|\|\s*\w+\s*;|`.*`)',
                    'file_inclusion': r'(?i)(include\(|require\(|include_once\(|require_once\()',
                }

                for attack_type, pattern in attack_patterns.items():
                    self.df[f'attack_{attack_type}'] = self.df['url'].str.contains(pattern, regex=True).astype(int)
                logger.info("Created features for attack patterns")

        except Exception as e:
            logger.error(f"Error in extracting HTTP features: {e}")
            raise

    def clean_data(self):
        """Execute the complete data cleaning pipeline."""
        logger.info("Starting data cleaning process...")
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Execute cleaning steps
        self.load_data()
        self.remove_duplicates()
        self.handle_missing_values()
        self.validate_data_types()
        self.standardize_connection_values()
        self.extract_http_features()
        
        # Save cleaned dataset
        output_file = self.output_dir / 'cleaned_dataset.csv'
        self.df.to_csv(output_file, index=False)
        logger.info(f"Cleaned dataset saved to {output_file}")
        
        return self.df

def main():
    # Define input and output paths
    input_file = "data/csic_database.csv"
    output_dir = "data/processed"
    
    # Initialize and run the cleaner
    cleaner = DataCleaner(input_file, output_dir)
    try:
        cleaned_data = cleaner.clean_data()
        logger.info("Data cleaning completed successfully")
        logger.info(f"Final dataset shape: {cleaned_data.shape}")
    except Exception as e:
        logger.error(f"Error during data cleaning: {e}")

if __name__ == "__main__":
    main()