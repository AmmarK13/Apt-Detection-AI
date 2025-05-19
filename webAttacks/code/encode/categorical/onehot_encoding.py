import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OneHotEncodingTransformer:
    def __init__(self, input_path):
        """Initialize the one-hot encoding transformer.

        Args:
            input_path (str): Path to the input CSV file
        """
        self.input_path = input_path
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    def transform(self, output_path):
        """Transform categorical features using one-hot encoding.

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
                logger.warning("No categorical columns found for one-hot encoding")
                df.to_csv(output_path, index=False)
                return

            # Fit and transform categorical columns
            encoded_data = self.encoder.fit_transform(df[categorical_columns])
            feature_names = self.encoder.get_feature_names_out(categorical_columns)

            # Create DataFrame with encoded features
            encoded_df = pd.DataFrame(
                encoded_data,
                columns=feature_names,
                index=df.index
            )

            # Combine with non-categorical columns
            numerical_columns = df.select_dtypes(exclude=['object']).columns
            result_df = pd.concat([df[numerical_columns], encoded_df], axis=1)

            # Save the transformed data
            result_df.to_csv(output_path, index=False)
            logger.info(f"Saved one-hot encoded data to {output_path}")

        except Exception as e:
            logger.error(f"Error during one-hot encoding transformation: {e}")
            raise