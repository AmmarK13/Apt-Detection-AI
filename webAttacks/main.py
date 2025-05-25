import logging
from pathlib import Path
from code.pip.pipeline_manager import PipelineManager
from code.pip.data_cleaning_step import DataCleaningStep
from code.pip.hybrid_encoding_step import HybridEncodingStep
from code.pip.minmax_scaler_step import MinMaxScalerStep
from code.pip.model_step import ModelStep
from code.pip.data_transformer_step import DataTransformerStep

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Initialize the pipeline manager
    pipeline = PipelineManager()
    
    # Add Data Cleaning step
    cleaning_step = DataCleaningStep(input_path="D:/University/Software Engineering/Project/Output/generated_dataset.csv")
    pipeline.add_step(cleaning_step.execute, "Data Cleaning")
    
    # Add Data Transformer step
    transformer_step = DataTransformerStep()
    pipeline.add_step(transformer_step.execute, "Data Transformation")
    
    # Add Hybrid encoding step
    hybrid_step = HybridEncodingStep()
    pipeline.add_step(hybrid_step.execute, "Hybrid Encoding")
    
    # Add MinMaxScaler step
    minmax_scaler_step = MinMaxScalerStep()   
    pipeline.add_step(minmax_scaler_step.execute, "MinMax Scaling")
    
    # Add Model evaluation step
    model_step = ModelStep()
    pipeline.add_step(model_step.execute, "Model Evaluation")
    
    # Execute the pipeline
    try:
        pipeline.run()
        logger.info("Data processing pipeline completed successfully")
    except Exception as e:
        logger.error(f"Error in pipeline execution: {e}")
        raise

if __name__ == "__main__":
    main()