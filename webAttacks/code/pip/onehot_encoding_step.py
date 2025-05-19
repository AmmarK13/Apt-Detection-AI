import logging
from pathlib import Path
from code.encode.categorical.onehot_encoding import OneHotEncodingTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OneHotEncodingStep:
    def __init__(self):
        """Initialize the one-hot encoding step."""
        self.input_path = "D:/University/Software Engineering/Project/data/csic_cleaned.csv"
        self.output_path = "D:/University/Software Engineering/Project/data/csic_onehotEncoded.csv"

    def execute(self):
        """Execute the one-hot encoding transformation step."""
        try:
            transformer = OneHotEncodingTransformer(self.input_path)
            transformer.transform(self.output_path)
            logger.info(f"One-hot encoding completed: {self.output_path}")
        except Exception as e:
            logger.error(f"Error in one-hot encoding step: {e}")
            raise