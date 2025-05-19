import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StandardScalerTransformer:
    def __init__(self, input_path):
        """Initialize the standard scaler transformer.

        Args:
            input_path (str): Path to the input CSV file
        """
        self.input_path = input_path
        self.scaler = StandardScaler()
        
    def transform(self, output_path):
        """Transform numerical features using standard scaling approach.

        Args:
            output_path (str): Path to save the transformed CSV file
        """
        try:
            # Read the input data
            df = pd.read_csv(self.input_path)
            logger.info(f"Loaded data from {self.input_path}")

            # Identify numerical columns (int64 and float64 dtypes)
            numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
            logger.info(f"Identified numerical columns: {numerical_columns.tolist()}")

            if len(numerical_columns) == 0:
                logger.warning("No numerical columns found for scaling")
                df.to_csv(output_path, index=False)
                return

            # Apply Standard Scaling to numerical columns
            scaled_data = self.scaler.fit_transform(df[numerical_columns])
            
            # Create DataFrame with scaled features
            scaled_df = pd.DataFrame(
                scaled_data,
                columns=numerical_columns,
                index=df.index
            )
            
            # Replace original numerical columns with scaled data
            df[numerical_columns] = scaled_df

            # Save the transformed data
            df.to_csv(output_path, index=False)
            logger.info(f"Saved standard scaled data to {output_path}")

        except Exception as e:
            logger.error(f"Error during standard scaling transformation: {e}")
            raise