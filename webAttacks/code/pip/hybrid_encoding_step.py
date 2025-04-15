import logging
from pathlib import Path
from code.encode.categorical.hybrid_encoding import HybridEncodingTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HybridEncodingStep:
    def __init__(self, cardinality_threshold=10):
        """Initialize the hybrid encoding step.

        Args:
            cardinality_threshold (int): Threshold for determining high vs low cardinality
        """
        self.input_path = "data/csic_database.csv"
        self.output_path = "data/csic_reduced_hybridEncoded.csv"
        self.cardinality_threshold = cardinality_threshold

    def execute(self):
        """Execute the hybrid encoding transformation step."""
        try:
            transformer = HybridEncodingTransformer(
                self.input_path,
                cardinality_threshold=self.cardinality_threshold
            )
            transformer.transform(self.output_path)
            logger.info(f"Hybrid encoding completed: {self.output_path}")
        except Exception as e:
            logger.error(f"Error in hybrid encoding step: {e}")
            raise