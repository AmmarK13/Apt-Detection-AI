import pandas as pd
import logging
import numpy as np
from pathlib import Path

class DataTransformer:
    def __init__(self):
        self.required_columns = [
            'cookie', 'url', 'method_get', 'method_post', 'method_put',
            'host_localhost:8080', 'host_localhost:9090', 'classification'
        ]
        self.logger = logging.getLogger(__name__)

    def transform_data(self, input_path, output_path):
        try:
            # Read the input file
            df = pd.read_csv(input_path)
            transformed_df = pd.DataFrame()
            
            # Handle cookie - convert to binary (0/1)
            if 'cookie' in df.columns:
                transformed_df['cookie'] = df['cookie'].notna().astype(int)
            else:
                self.logger.warning("Missing cookie column. Setting to 0.")
                transformed_df['cookie'] = 0
            
            # Handle URL - convert to length as numerical feature
            if 'URL' in df.columns:
                # First ensure URLs have protocol
                urls = df['URL'].fillna('')
                urls = urls.apply(
                    lambda x: f"http://{x}" if x and not x.startswith(('http://', 'https://')) else x
                )
                # Convert to numerical feature (URL length)
                transformed_df['URL'] = urls.str.len().astype(float)
            else:
                self.logger.warning("Missing URL column. Setting to 0.")
                transformed_df['URL'] = 0.0
            
            # Handle method columns - already numerical (0/1)
            if 'Method' in df.columns:
                transformed_df['Method_GET'] = (df['Method'].str.upper() == 'GET').astype(int)
                transformed_df['Method_POST'] = (df['Method'].str.upper() == 'POST').astype(int)
                transformed_df['Method_PUT'] = (df['Method'].str.upper() == 'PUT').astype(int)
                
                method_sum = transformed_df[['Method_GET', 'Method_POST', 'Method_PUT']].sum(axis=1)
                if not (method_sum == 1).all():
                    self.logger.warning("Invalid method values found. Defaulting to GET.")
                    invalid_rows = method_sum != 1
                    transformed_df.loc[invalid_rows, 'Method_GET'] = 1
                    transformed_df.loc[invalid_rows, 'Method_POST'] = 0
                    transformed_df.loc[invalid_rows, 'Method_PUT'] = 0
            else:
                transformed_df['Method_GET'] = 1
                transformed_df['Method_POST'] = 0
                transformed_df['Method_PUT'] = 0
            
            # Handle host columns - already numerical (0/1)
            if 'host' in df.columns:
                transformed_df['host_localhost:8080'] = (df['host'] == 'localhost:8080').astype(int)
                transformed_df['host_localhost:9090'] = (df['host'] == 'localhost:9090').astype(int)
                
                host_sum = transformed_df[['host_localhost:8080', 'host_localhost:9090']].sum(axis=1)
                if not (host_sum == 1).all():
                    invalid_rows = host_sum != 1
                    transformed_df.loc[invalid_rows, 'host_localhost:8080'] = 1
                    transformed_df.loc[invalid_rows, 'host_localhost:9090'] = 0
            else:
                transformed_df['host_localhost:8080'] = 1
                transformed_df['host_localhost:9090'] = 0
            
            # Handle classification
            if 'classification' in df.columns:
                transformed_df['classification'] = df['classification'].astype(int)
            
            # Ensure all columns are numerical
            for col in transformed_df.columns:
                if not np.issubdtype(transformed_df[col].dtype, np.number):
                    self.logger.warning(f"Converting column {col} to numeric")
                    transformed_df[col] = pd.to_numeric(transformed_df[col], errors='coerce').fillna(0)
            
            # Update required columns
            self.required_columns = [
                'cookie', 'URL', 'Method_GET', 'Method_POST', 'Method_PUT',
                'host_localhost:8080', 'host_localhost:9090', 'classification'
            ]
            
            # Ensure column order
            transformed_df = transformed_df[self.required_columns]
            
            # Save transformed data
            transformed_df.to_csv(output_path, index=False)
            self.logger.info(f"Transformed data saved to: {output_path}")
            
            return transformed_df
            
        except Exception as e:
            self.logger.error(f"Error transforming data: {e}")
            raise