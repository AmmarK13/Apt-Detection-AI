import pandas as pd
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EncodingValidator:
    def __init__(self):
        """Initialize the encoding validator."""
        pass

    def validate_unique_mapping(self, original_df, encoded_df, column_name):
        """Validate that each unique value in the original column maps to a unique number.

        Args:
            original_df (pd.DataFrame): Original dataframe before encoding
            encoded_df (pd.DataFrame): Encoded dataframe after transformation
            column_name (str): Name of the column to validate

        Returns:
            bool: True if mapping is unique, False otherwise
        """
        try:
            # Create mapping dictionary
            value_to_encoding = {}
            encoding_to_value = {}

            # Iterate through both dataframes simultaneously
            for orig_val, encoded_val in zip(original_df[column_name], encoded_df[column_name]):
                # Check if original value already maps to a different encoding
                if orig_val in value_to_encoding:
                    if value_to_encoding[orig_val] != encoded_val:
                        logger.error(f"Value '{orig_val}' maps to multiple encodings: "
                                    f"{value_to_encoding[orig_val]} and {encoded_val}")
                        return False
                # Check if encoding already maps to a different original value
                if encoded_val in encoding_to_value:
                    if encoding_to_value[encoded_val] != orig_val:
                        logger.error(f"Encoding {encoded_val} maps to multiple values: "
                                    f"'{encoding_to_value[encoded_val]}' and '{orig_val}'")
                        return False
                
                # Store mappings
                value_to_encoding[orig_val] = encoded_val
                encoding_to_value[encoded_val] = orig_val

            logger.info(f"Validated unique one-to-one mapping for column: {column_name}")
            return True

        except Exception as e:
            logger.error(f"Error during encoding validation: {e}")
            raise