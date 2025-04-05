import pandas as pd
import numpy as np
from urllib.parse import urlparse, parse_qs
import re

class FeatureEngineer:
    def __init__(self):
        """
        Initialize the FeatureEngineer with predefined suspicious patterns for attack detection.
        Patterns include SQL injection, XSS, path traversal, command injection, and special characters.
        """
        self.suspicious_patterns = {
            'sql_injection': r'(\bunion\b|\bselect\b|\bfrom\b|\bdrop\b|--|\%27|\'|\#)',
            'xss': r'(<script>|alert\(|eval\(|javascript:|<img|onload=)',
            'path_traversal': r'(\.\.\/|\.\.\\|\%2e\%2e)',
            'command_injection': r'(;|\||`|\$\(|\&\&|\|\|)',
            'special_chars': r'(!|\$|#|%|&|\*)',
        }

    def extract_url_features(self, df):
        """
        Extract features from URL components:
        - url_length: Total length of the URL
        - path_depth: Number of directories in the URL path
        - query_length: Length of the query string
        - param_count: Number of parameters in the query
        - contains_*: Binary indicators for suspicious patterns in URL
        
        Args:
            df (DataFrame): Input dataframe with 'URL' column
        Returns:
            DataFrame: Dataframe with additional URL-based features
        """
        df['url_length'] = df['URL'].str.len()
        
        # Parse URL components
        parsed_urls = df['URL'].apply(urlparse)
        df['path_depth'] = parsed_urls.apply(lambda x: len(x.path.split('/')))
        df['query_length'] = parsed_urls.apply(lambda x: len(x.query))
        df['param_count'] = parsed_urls.apply(lambda x: len(parse_qs(x.query)))
        
        # Detect suspicious patterns
        for pattern_name, pattern in self.suspicious_patterns.items():
            df[f'contains_{pattern_name}'] = df['URL'].str.contains(pattern, regex=True).astype(int)
        
        return df

    def analyze_content(self, df):
        """
        Extract features from request content:
        - content_length: Length of the content
        - content_special_chars: Count of special characters
        - content_numbers: Count of numerical characters
        - has_base64: Binary indicator for base64 encoded content
        - has_hex_encoding: Binary indicator for hex encoded content
        
        Args:
            df (DataFrame): Input dataframe with 'content' column
        Returns:
            DataFrame: Dataframe with additional content-based features
        """
        df['content_length'] = df['content'].fillna('').str.len()
        df['content_special_chars'] = df['content'].fillna('').str.count(r'[!@#$%^&*(),.?":{}|<>]')
        df['content_numbers'] = df['content'].fillna('').str.count(r'\d')
        df['has_base64'] = df['content'].fillna('').str.contains(r'^[A-Za-z0-9+/=]+$').astype(int)
        df['has_hex_encoding'] = df['content'].fillna('').str.contains(r'%[0-9A-Fa-f]{2}').astype(int)
        return df

    def encode_categorical(self, df):
        """
        Encode categorical variables using one-hot encoding:
        - HTTP methods (GET, POST, etc.)
        - Content type categories (text, application, multipart, other)
        
        Args:
            df (DataFrame): Input dataframe with categorical columns
        Returns:
            DataFrame: Dataframe with encoded categorical features
        """
        # One-hot encoding for HTTP methods
        df = pd.get_dummies(df, columns=['Method'], prefix='method')
        
        # Encode content types
        df['content_type_category'] = df['content-type'].fillna('unknown').apply(
            lambda x: 'text' if 'text' in x.lower() 
            else 'application' if 'application' in x.lower()
            else 'multipart' if 'multipart' in x.lower()
            else 'other'
        )
        df = pd.get_dummies(df, columns=['content_type_category'], prefix='content_type')
        
        return df

    def extract_user_agent_features(self, df):
        """
        Extract features from User-Agent string:
        - is_mobile: Binary indicator for mobile devices
        - is_bot: Binary indicator for bot/crawler requests
        - browser_type: One-hot encoded browser categories
        
        Args:
            df (DataFrame): Input dataframe with 'User-Agent' column
        Returns:
            DataFrame: Dataframe with additional User-Agent based features
        """
        df['is_mobile'] = df['User-Agent'].str.contains(r'Mobile|Android|iPhone', case=False).astype(int)
        df['is_bot'] = df['User-Agent'].str.contains(r'bot|crawler|spider', case=False).astype(int)
        df['browser_type'] = df['User-Agent'].apply(
            lambda x: 'chrome' if 'chrome' in str(x).lower()
            else 'firefox' if 'firefox' in str(x).lower()
            else 'ie' if 'msie' in str(x).lower()
            else 'other'
        )
        df = pd.get_dummies(df, columns=['browser_type'], prefix='browser')
        return df

    def generate_features(self, input_file, output_file):
        """
        Main feature generation pipeline:
        1. Load the cleaned dataset
        2. Extract URL-based features
        3. Extract content-based features
        4. Encode categorical variables
        5. Extract User-Agent features
        6. Save the featured dataset
        
        Args:
            input_file (str): Path to input CSV file
            output_file (str): Path to save featured CSV file
        Returns:
            DataFrame: Complete dataframe with all generated features
        """
        print("Loading data...")
        df = pd.read_csv(input_file)
        
        print("Extracting URL features...")
        df = self.extract_url_features(df)
        
        print("Analyzing content...")
        df = self.analyze_content(df)
        
        print("Encoding categorical features...")
        df = self.encode_categorical(df)
        
        print("Extracting User-Agent features...")
        df = self.extract_user_agent_features(df)
        
        print(f"Saving featured dataset to {output_file}")
        df.to_csv(output_file, index=False)
        
        print(f"Generated {len(df.columns)} features")
        return df

if __name__ == "__main__":
    engineer = FeatureEngineer()
    featured_df = engineer.generate_features(
        "data/cleaned_csic_database.csv",
        "data/featured_csic_database.csv"
    )