import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HybridEncodingTransformer:
    def __init__(self, input_path, cardinality_threshold=10):
        """Initialize the hybrid encoding transformer.

        Args:
            input_path (str): Path to the input CSV file
            cardinality_threshold (int): Threshold for determining high vs low cardinality
        """
        self.input_path = input_path
        self.cardinality_threshold = cardinality_threshold
        self.label_encoder = {}
        self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
    def _get_cardinality(self, column):
        """Get the cardinality (number of unique values) of a column."""
        return len(column.unique())
    
    def transform(self, output_path):
        """Transform categorical features using hybrid encoding approach.

        Args:
            output_path (str): Path to save the transformed CSV file
        """
        try:
            # Read the input data
            df = pd.read_csv(self.input_path)
            logger.info(f"Loaded data from {self.input_path}")

            # Identify categorical columns (object dtype)
            categorical_columns = df.select_dtypes(include=['object']).columns
            logger.info(f"Identified categorical columns: {categorical_columns.tolist()}")

            if len(categorical_columns) == 0:
                logger.warning("No categorical columns found for encoding")
                df.to_csv(output_path, index=False)
                return

            # Separate high and low cardinality columns
            high_cardinality = []
            low_cardinality = []
            
            for col in categorical_columns:
                cardinality = self._get_cardinality(df[col])
                if cardinality > self.cardinality_threshold:
                    high_cardinality.append(col)
                else:
                    low_cardinality.append(col)
                    
            logger.info(f"High cardinality columns: {high_cardinality}")
            logger.info(f"Low cardinality columns: {low_cardinality}")

            # Apply Label Encoding to high cardinality columns
            for col in high_cardinality:
                self.label_encoder[col] = LabelEncoder()
                df[col] = self.label_encoder[col].fit_transform(df[col])

            # Apply One-Hot Encoding to low cardinality columns
            if low_cardinality:
                encoded_data = self.onehot_encoder.fit_transform(df[low_cardinality])
                feature_names = self.onehot_encoder.get_feature_names_out(low_cardinality)
                
                # Create DataFrame with encoded features
                encoded_df = pd.DataFrame(
                    encoded_data,
                    columns=feature_names,
                    index=df.index
                )
                
                # Drop original low cardinality columns and combine with encoded data
                df = df.drop(columns=low_cardinality)
                df = pd.concat([df, encoded_df], axis=1)

            # Save the transformed data
            df.to_csv(output_path, index=False)
            logger.info(f"Saved hybrid encoded data to {output_path}")

        except Exception as e:
            logger.error(f"Error during hybrid encoding transformation: {e}")
            raise