import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from .base import PipelineStage
import logging

logger = logging.getLogger(__name__)

class DataTransformer(PipelineStage):
    """Handles data transformation operations including encoding and scaling."""

    def __init__(self, input_path: str, output_path: str):
        super().__init__(input_path, output_path)
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical columns using LabelEncoder."""
        categorical_columns = df.select_dtypes(include=['object']).columns
        df_encoded = df.copy()

        for column in categorical_columns:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
            df_encoded[column] = self.label_encoders[column].fit_transform(df[column])

        return df_encoded

    def scale_numerical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features using StandardScaler."""
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
        df_scaled = df.copy()
        df_scaled[numerical_columns] = self.scaler.fit_transform(df[numerical_columns])
        return df_scaled

    def validate(self):
        """Validate input data existence and format."""
        try:
            df = pd.read_csv(self.input_path)
            if df.empty:
                raise ValueError('Input data is empty')
            logger.info('Data validation successful')
        except Exception as e:
            logger.error(f'Data validation failed: {e}')
            raise

    def process(self) -> pd.DataFrame:
        """Process the input data with encoding and scaling."""
        try:
            df = pd.read_csv(self.input_path)
            
            # Handle missing values
            df = df.fillna(df.mean())
            
            # Transform categorical features
            df = self.encode_categorical(df)
            
            # Scale numerical features
            df = self.scale_numerical(df)
            
            # Save transformed data
            df.to_csv(self.output_path, index=False)
            logger.info(f'Transformed data saved to {self.output_path}')
            
            return df
        except Exception as e:
            logger.error(f'Error during data transformation: {e}')
            raise