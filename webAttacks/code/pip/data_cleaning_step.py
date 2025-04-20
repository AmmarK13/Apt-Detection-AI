import pandas as pd
import logging
from pathlib import Path
from code.cleaning.csic_cleaned import clean_csic_dataset

logger = logging.getLogger(__name__)

class DataCleaningStep:
    def __init__(self):
        self.input_path = "D:/University/Software Engineering/Project/Apt-Detection-AI/webAttacks/data/csic_database.csv"
        self.output_path = "D:/University/Software Engineering/Project/Apt-Detection-AI/webAttacks/data/csic_cleaned.csv"

    def execute(self):
        try:
            # Create output directory if it doesn't exist
            output_dir = Path(self.output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Clean the dataset
            cleaned_df = clean_csic_dataset(self.input_path)
            
            # Save cleaned dataset
            cleaned_df.to_csv(self.output_path, index=False)
            logger.info(f"Cleaned dataset saved to {self.output_path}")
            
        except Exception as e:
            logger.error(f"Error in data cleaning step: {e}")
            raise