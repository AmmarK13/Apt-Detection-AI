import logging
from typing import List, Callable
from pathlib import Path
from code.pip.hybrid_encoding_step import HybridEncodingStep

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PipelineManager:
    def __init__(self):
        """Initialize the pipeline manager."""
        self.steps = []
        self.step_names = []

    def add_step(self, step_func: Callable, name: str = None):
        """Add a processing step to the pipeline.

        Args:
            step_func (Callable): The function to execute for this step
            name (str, optional): Name of the step for logging purposes
        """
        if name is None:
            name = step_func.__name__
        self.steps.append(step_func)
        self.step_names.append(name)
        logger.info(f"Added pipeline step: {name}")

    def run(self):
        """Execute all steps in the pipeline sequentially."""
        logger.info("Starting pipeline execution")
        
        try:
            # Hybrid encoding step (combines label and one-hot encoding)
            hybrid_encoding_step = HybridEncodingStep(cardinality_threshold=10)
            hybrid_encoding_step.execute()
            
            for step_func, step_name in zip(self.steps, self.step_names):
                logger.info(f"Executing step: {step_name}")
                step_func()
                logger.info(f"Completed step: {step_name}")

            logger.info("Pipeline execution completed successfully")
            
        except Exception as e:
            logger.error(f"Error in pipeline execution at step {step_name}: {e}")
            raise