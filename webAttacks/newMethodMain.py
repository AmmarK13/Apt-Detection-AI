import logging
from pathlib import Path
from code.pip.pipeline_manager import PipelineManager
from code.pip.label_encoding_step import LabelEncodingStep

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Initialize the pipeline manager
    pipeline = PipelineManager()
    
    # Add processing steps in the desired order
    label_encoding_step = LabelEncodingStep()
    pipeline.add_step(label_encoding_step.execute, "Label Encoding")
    
    # Execute the pipeline
    try:
        pipeline.run()
        logger.info("Data processing pipeline completed successfully")
    except Exception as e:
        logger.error(f"Error in pipeline execution: {e}")
        raise

if __name__ == "__main__":
    main()