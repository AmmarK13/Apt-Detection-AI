import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StandardScalerStep:
    def __init__(self):
        self.input_path = "D:/University/Software Engineering/Project/data/csic_reduced_hybridEncoded.csv"
        self.output_path = "D:/University/Software Engineering/Project/data/csic_reduced_StandardScaledEncoded.csv"

    def execute(self):
        """Execute the standard scaling transformation step."""
        try:
            # Read the data
            df = pd.read_csv(self.input_path)
            
            # Separate features and target
            X = df.drop('classification', axis=1)
            y = df['classification']
            
            # Apply StandardScaler only to features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Convert scaled features back to DataFrame
            X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
            
            # Combine scaled features with target
            final_df = pd.concat([X_scaled_df, y], axis=1)
            
            # Save the scaled data
            final_df.to_csv(self.output_path, index=False)
            logger.info(f"Standard scaling completed: {self.output_path}")
            
        except Exception as e:
            logger.error(f"Error in standard scaling step: {e}")
            raise