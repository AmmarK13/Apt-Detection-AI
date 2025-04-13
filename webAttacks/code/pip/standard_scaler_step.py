import logging
from pathlib import Path
from code.encode.numerical.standard_scaler import StandardScalerTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StandardScalerStep:
    def __init__(self):
        """Initialize the standard scaler step."""
        self.input_path = "D:/University/Software Engineering/Project/data/csic_hybridEncoded.csv"
        self.output_path = "D:/University/Software Engineering/Project/data/csic_standardScaledEncoded.csv"

    def execute(self):
        """Execute the standard scaling transformation step."""
        try:
            transformer = StandardScalerTransformer(self.input_path)
            transformer.transform(self.output_path)
            logger.info(f"Standard scaling completed: {self.output_path}")
        except Exception as e:
            logger.error(f"Error in standard scaling step: {e}")
            raise