import pandas as pd
import numpy as np
from typing import List, Dict
from .base import PipelineStage
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer(PipelineStage):
    """Handles feature engineering for web attack detection."""

    def __init__(self, input_path: str, output_path: str):
        super().__init__(input_path, output_path)
        self.feature_stats = {}

    def extract_url_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from URL patterns."""
        df_features = df.copy()
        
        # URL length features
        df_features['url_length'] = df['url'].str.len()
        df_features['path_length'] = df['url'].str.split('?').str[0].str.len()
        
        # Special character counts
        df_features['special_char_count'] = df['url'].str.count('[^a-zA-Z0-9]')
        df_features['param_count'] = df['url'].str.count('=')
        
        return df_features

    def extract_payload_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from request payload."""
        df_features = df.copy()
        
        # Payload characteristics
        df_features['payload_length'] = df['payload'].str.len()
        df_features['script_tags'] = df['payload'].str.count('<script')
        df_features['sql_keywords'] = df['payload'].str.count('(?i)(select|union|insert|delete|update)')
        
        return df_features

    def generate_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate statistical features from numeric columns."""
        df_features = df.copy()
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        # Calculate basic statistics
        for col in numeric_cols:
            self.feature_stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max()
            }
            
            # Generate normalized features
            df_features[f'{col}_normalized'] = (df[col] - self.feature_stats[col]['mean']) / self.feature_stats[col]['std']
        
        return df_features

    def validate(self):
        """Validate input data and required columns."""
        try:
            df = pd.read_csv(self.input_path)
            required_columns = ['url', 'payload']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f'Missing required columns: {missing_columns}')
                
            logger.info('Feature engineering validation successful')
        except Exception as e:
            logger.error(f'Feature engineering validation failed: {e}')
            raise

    def process(self) -> pd.DataFrame:
        """Process the input data and engineer features."""
        try:
            df = pd.read_csv(self.input_path)
            
            # Apply feature engineering steps
            df = self.extract_url_features(df)
            df = self.extract_payload_features(df)
            df = self.generate_statistical_features(df)
            
            # Save engineered features
            df.to_csv(self.output_path, index=False)
            logger.info(f'Engineered features saved to {self.output_path}')
            
            return df
        except Exception as e:
            logger.error(f'Error during feature engineering: {e}')
            raise