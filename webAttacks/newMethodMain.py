import logging
from pathlib import Path
from code.pip.pipeline_manager import PipelineManager
from code.pip.label_encoding_step import LabelEncodingStep
from code.pip.onehot_encoding_step import OneHotEncodingStep
from code.pip.hybrid_encoding_step import HybridEncodingStep
from code.pip.minmax_scaler_step import MinMaxScalerStep

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Initialize the pipeline manager
    pipeline = PipelineManager()
    # Add Hybrid encoding step
    # Hybrid Encoding combines both Label and OneHot encoding techniques
    # For cookies: Uses Label encoding for categorical values and patterns
    #   - Encodes common cookie patterns (e.g., session IDs, authentication tokens)
    #   - Converts cookie values into numerical representations
    # For URLs: Uses OneHot encoding for critical components
    #   - Encodes URL paths, query parameters, and special characters
    #   - Creates binary features for common attack patterns
    #   - Preserves URL structure while converting to machine-readable format
    hybrid_step = HybridEncodingStep()
    pipeline.add_step(hybrid_step.execute, "Hybrid Encoding")
    # Add MinMaxScaler step
    minmax_scaler_step = MinMaxScalerStep()   
    pipeline.add_step(minmax_scaler_step.execute, "MinMax Scaling")
    
    # Execute the pipeline
    try:
        pipeline.run()
        logger.info("Data processing pipeline completed successfully")
    except Exception as e:
        logger.error(f"Error in pipeline execution: {e}")
        raise

if __name__ == "__main__":
    main()










    # Add OneHot encoding step
    # onehot_step = OneHotEncodingStep()
    # pipeline.add_step(onehot_step.execute, "OneHot Encoding")
    # # Add processing steps in the desired order
    # label_encoding_step = LabelEncodingStep()
    # pipeline.add_step(label_encoding_step.execute, "Label Encoding")