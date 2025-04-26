from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class PipelineStage(ABC):
    """Abstract base class for all pipeline stages."""
    
    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path

    @abstractmethod
    def process(self):
        """Process the input data and generate output."""
        pass

    @abstractmethod
    def validate(self):
        """Validate the input and output data."""
        pass

    def run(self):
        """Execute the pipeline stage with validation."""
        try:
            logger.info(f'Starting {self.__class__.__name__}')
            self.validate()
            result = self.process()
            logger.info(f'Completed {self.__class__.__name__}')
            return result
        except Exception as e:
            logger.error(f'Error in {self.__class__.__name__}: {e}')
            raise