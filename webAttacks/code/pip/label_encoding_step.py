import logging
from pathlib import Path
from code.encode.categorical.label_encoding import LabelEncodingTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LabelEncodingStep:
    def __init__(self):
        """Initialize the label encoding step."""
        self.input_path = "D:/University/Software Engineering/Project/data/csic_cleaned.csv"
        self.output_path = "D:/University/Software Engineering/Project/data/csic_labelEncoded.csv"

    def execute(self):
        """Execute the label encoding transformation step."""
        try:
            transformer = LabelEncodingTransformer(self.input_path)
            transformer.transform(self.output_path)
            logger.info(f"Label encoding completed: {self.output_path}")
        except Exception as e:
            logger.error(f"Error in label encoding step: {e}")
            raise