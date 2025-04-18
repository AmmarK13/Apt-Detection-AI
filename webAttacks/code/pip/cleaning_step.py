import logging
from pathlib import Path
from src.cleaning.clean_data import DataCleaner
from .base import PipelineStep

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CleaningStep(PipelineStep):
    def __init__(self, input_path: str, output_path: str):
        """Initialize the cleaning step.

        Args:
            input_path (str): Path to the input CSV file
            output_path (str): Path to save the cleaned CSV file
        """
        super().__init__(input_path, output_path)
        self.cleaner = DataCleaner(input_path, Path(output_path).parent)

    def transform(self):
        """Execute the data cleaning transformation."""
        try:
            logger.info(f"Starting cleaning step with input: {self.input_path}")
            
            # Run the cleaning pipeline
            cleaned_data = self.cleaner.clean_data()
            
            # Ensure the output directory exists
            output_dir = Path(self.output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the cleaned data
            cleaned_data.to_csv(self.output_path, index=False)
            logger.info(f"Cleaning step completed. Output saved to: {self.output_path}")
            
        except Exception as e:
            logger.error(f"Error in cleaning step: {e}")
            raise