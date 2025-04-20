import logging
from pathlib import Path
from code.model.model import test_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelStep:
    def __init__(self):
        """Initialize the Model evaluation step."""
        self.input_path = "D:/University/Software Engineering/Project/data/minmaxScaled.csv"
        self.has_labels = True  # Set to False if your data doesn't have 'classification' column

    def execute(self):
        """Execute the model evaluation step."""
        try:
            logger.info(f"Starting model evaluation on: {self.input_path}")
            results = test_model(self.input_path, self.has_labels)
            logger.info(f"Model evaluation completed successfully")
            return results
        except Exception as e:
            logger.error(f"Error in model evaluation step: {e}")
            raise