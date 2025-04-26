import logging
from pathlib import Path
from ..transform.data_transformer import DataTransformer

class DataTransformerStep:
    def __init__(self):
        self.input_path = "D:/University/Software Engineering/Project/Output/cleaned.csv"
        self.output_path = "D:/University/Software Engineering/Project/Output/transformed.csv"
        self.transformer = DataTransformer()
        self.logger = logging.getLogger(__name__)

    def execute(self, data=None):
        """
        Execute the data transformation step
        
        Args:
            data: Optional input data path or DataFrame
            
        Returns:
            DataFrame: Transformed data ready for model
        """
        try:
            # Determine input path
            input_path = data if isinstance(data, str) else self.input_path
            
            # Create output directory if it doesn't exist
            output_dir = Path(self.output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Transform the data
            transformed_data = self.transformer.transform_data(input_path, self.output_path)
            
            self.logger.info(f"Data transformation completed. Output saved to {self.output_path}")
            return transformed_data
            
        except Exception as e:
            self.logger.error(f"Error in data transformation step: {e}")
            raise