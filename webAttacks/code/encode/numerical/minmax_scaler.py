import logging
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MinMaxScalerTransformer:
    def __init__(self, input_path):
        """Initialize the MinMax scaler transformer."""
        self.input_path = input_path
        self.scaler = MinMaxScaler()
        
    def transform(self, output_path):
        """Transform numerical features using MinMax scaling."""
        try:
            # Read the input data
            df = pd.read_csv(self.input_path)
            
            # Identify numerical columns
            numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
            
            if len(numerical_columns) > 0:
                # Scale numerical features
                scaled_data = self.scaler.fit_transform(df[numerical_columns])
                
                # Round to 2 decimal places
                scaled_data = pd.DataFrame(scaled_data, columns=numerical_columns).round(2)
                
                # Replace original numerical columns with scaled values
                df[numerical_columns] = scaled_data
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save the transformed data
            df.to_csv(output_path, index=False)
            logger.info(f"MinMax scaling completed and saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error in MinMax scaling transformation: {e}")
            raise