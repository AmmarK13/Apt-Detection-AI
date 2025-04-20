import pandas as pd
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clean_csic_dataset(input_file):
    """Clean the CSIC dataset by removing duplicates, unnamed columns, and handling missing values.
    
    Args:
        input_file (str): Path to the input CSV file
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    try:
        # Load dataset
        logger.info(f"Loading dataset from {input_file}")
        df = pd.read_csv(input_file)
        
        # Store original shape
        original_shape = df.shape
        logger.info(f"Original dataset shape: {original_shape}")
        
        # Drop unnamed columns
        unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)
            logger.info(f"Dropped unnamed columns: {unnamed_cols}")
        
        # Remove duplicates
        df = df.drop_duplicates()
        logger.info(f"Removed {original_shape[0] - len(df)} duplicate rows")
        
        # Drop columns with single unique value
        single_value_cols = [col for col in df.columns if df[col].nunique() == 1]
        if single_value_cols:
            df = df.drop(columns=single_value_cols)
            logger.info(f"Dropped columns with single unique value: {single_value_cols}")
        
        # Calculate missing value percentages
        missing_ratio = df.isna().mean() * 100
        high_missing_cols = missing_ratio[missing_ratio > 70].index.tolist()
        
        if high_missing_cols:
            logger.info(f"Dropping columns with >70% missing values: {high_missing_cols}")
            df = df.drop(columns=high_missing_cols)
        
        # Handle remaining missing values
        df = df.fillna('')
        logger.info("Filled remaining missing values with empty strings")
        
        # Validate data types
        if 'length' in df.columns:
            df['length'] = pd.to_numeric(df['length'], errors='coerce').fillna(0).astype(int)
        
        if 'classification' in df.columns:
            df['classification'] = df['classification'].astype(int)
        
        logger.info(f"Final dataset shape: {df.shape}")
        logger.info(f"Final columns: {df.columns.tolist()}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error during data cleaning: {e}")
        raise

def main():
    try:
        # Define input and output paths
        input_file = "D:/University/Software Engineering/Project/Apt-Detection-AI/webAttacks/data/csic_database.csv"
        output_file = "D:/University/Software Engineering/Project/data/csic_cleaned.csv"
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_file).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean the dataset
        cleaned_df = clean_csic_dataset(input_file)
        
        # Save cleaned dataset
        cleaned_df.to_csv(output_file, index=False)
        logger.info(f"Cleaned dataset saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
