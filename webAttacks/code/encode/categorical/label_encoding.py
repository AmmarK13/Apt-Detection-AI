import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LabelEncodingTransformer:
    def __init__(self, input_path):
        """Initialize the label encoding transformer.

        Args:
            input_path (str): Path to the input CSV file
        """
        self.input_path = input_path
        self.label_encoders = {}
        self.df = None
        
    def load_data(self):
        """Load the data from CSV file."""
        try:
            self.df = pd.read_csv(self.input_path)
            logger.info(f"Successfully loaded data from {self.input_path}")
            logger.info(f"Dataset shape: {self.df.shape}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def identify_categorical_columns(self):
        """Identify categorical columns in the dataset."""
        try:
            # Get categorical columns (object dtype), excluding 'classification'
            categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
            if 'classification' in categorical_cols:
                categorical_cols.remove('classification')  # Exclude classification column
            logger.info(f"Identified {len(categorical_cols)} categorical columns: {categorical_cols}")
            return categorical_cols
        except Exception as e:
            logger.error(f"Error identifying categorical columns: {e}")
            raise

    def encode_categorical_features(self):
        """Encode categorical features using LabelEncoder."""
        try:
            categorical_cols = self.identify_categorical_columns()
            
            # Create and fit label encoders for each categorical column
            for col in categorical_cols:
                logger.info(f"Encoding column: {col}")
                le = LabelEncoder()
                # Handle missing values by filling with a placeholder
                self.df[col] = self.df[col].fillna('missing')
                # Fit and transform the column
                self.df[col] = le.fit_transform(self.df[col])
                # Store the encoder
                self.label_encoders[col] = le
                
            logger.info("Successfully encoded all categorical features")
        except Exception as e:
            logger.error(f"Error in categorical encoding: {e}")
            raise

    def save_encoded_data(self, output_path):
        """Save the encoded dataset to a CSV file.

        Args:
            output_path (str): Path where to save the encoded CSV file
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # Save the encoded dataset
            self.df.to_csv(output_path, index=False)
            logger.info(f"Successfully saved encoded data to {output_path}")
        except Exception as e:
            logger.error(f"Error saving encoded data: {e}")
            raise

    def transform(self, output_path):
        """Complete transformation pipeline.

        Args:
            output_path (str): Path where to save the encoded CSV file
        """
        try:
            self.load_data()
            self.encode_categorical_features()
            self.save_encoded_data(output_path)
            logger.info("Label encoding transformation completed successfully")
        except Exception as e:
            logger.error(f"Error in transformation pipeline: {e}")
            raise

def main():
    # Define input and output paths
    input_path = "D:/University/Software Engineering/Project/data/csic_cleaned.csv"
    output_path = "D:/University/Software Engineering/Project/data/csic_labelEncoded.csv"
    
    try:
        # Initialize and run the transformer
        transformer = LabelEncodingTransformer(input_path)
        transformer.transform(output_path)
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()