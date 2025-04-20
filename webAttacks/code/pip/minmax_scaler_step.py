import logging
from pathlib import Path
from code.encode.numerical.minmax_scaler import MinMaxScalerTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MinMaxScalerStep:
    def __init__(self):
        """Initialize the MinMax scaling step."""
        self.input_path = "D:/University/Software Engineering/Project/data/hybridEncoded.csv"
        self.output_path = "D:/University/Software Engineering/Project/data/minmaxScaled.csv"

    def execute(self):
        """Execute the MinMax scaling transformation step."""
        try:
            transformer = MinMaxScalerTransformer(self.input_path)
            transformer.transform(self.output_path)
            logger.info(f"MinMax scaling completed: {self.output_path}")
        except Exception as e:
            logger.error(f"Error in MinMax scaling step: {e}")
            raise