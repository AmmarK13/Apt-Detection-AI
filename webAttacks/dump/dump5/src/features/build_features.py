import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureBuilder:
    def __init__(self, input_file: str, output_dir: str):
        """Initialize the FeatureBuilder with input and output paths.

        Args:
            input_file (str): Path to the cleaned CSV file
            output_dir (str): Directory to save feature engineered data
        """
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.df = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.tfidf = TfidfVectorizer(max_features=200, ngram_range=(1, 3))
        self.count_vec = CountVectorizer(max_features=100, ngram_range=(1, 2))

    def load_data(self):
        """Load the cleaned dataset."""
        try:
            self.df = pd.read_csv(self.input_file)
            logger.info(f"Loaded cleaned dataset with shape: {self.df.shape}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def extract_text_features(self):
        """Extract advanced features from text data using TF-IDF and additional patterns."""
        try:
            # Extract features from URL parameters if available
            if 'url' in self.df.columns:
                # TF-IDF features with increased max_features and n-gram range
                self.tfidf = TfidfVectorizer(max_features=300, ngram_range=(1, 4))
                url_features = self.tfidf.fit_transform(self.df['url'].fillna(''))
                url_feature_names = [f'url_tfidf_{i}' for i in range(url_features.shape[1])]
                url_features_df = pd.DataFrame(
                    url_features.toarray(),
                    columns=url_feature_names
                )
                
                # Enhanced count vectorizer features
                self.count_vec = CountVectorizer(max_features=150, ngram_range=(1, 3))
                count_features = self.count_vec.fit_transform(self.df['url'].fillna(''))
                count_feature_names = [f'url_count_{i}' for i in range(count_features.shape[1])]
                count_features_df = pd.DataFrame(
                    count_features.toarray(),
                    columns=count_feature_names
                )
                
                # Advanced character-based features
                self.df['special_char_ratio'] = self.df['url'].fillna('').apply(
                    lambda x: len(re.findall(r'[^a-zA-Z0-9\s]', x)) / (len(x) + 1)
                )
                self.df['numeric_ratio'] = self.df['url'].fillna('').apply(
                    lambda x: len(re.findall(r'[0-9]', x)) / (len(x) + 1)
                )
                self.df['uppercase_ratio'] = self.df['url'].fillna('').apply(
                    lambda x: len(re.findall(r'[A-Z]', x)) / (len(x) + 1)
                )
                
                # Enhanced SQL injection detection
                sql_patterns = (
                    r'\b(select|union|insert|delete|from|drop|update|where|having|group\s+by)\b|'
                    r'[\"\']\s*or\s*[\"\']|'
                    r'\b(and|or)\s+\d+\s*=\s*\d+|'
                    r'\b(sleep|benchmark|wait)\s*\(|'
                    r'--\s*$|#\s*$'
                )
                self.df['sql_indicator'] = self.df['url'].fillna('').apply(
                    lambda x: int(bool(re.search(sql_patterns, x.lower())))
                )
                
                # Enhanced XSS attack detection
                xss_patterns = (
                    r'(<script|javascript:|on\w+\s*=|alert\s*\()|'
                    r'(\\x[0-9a-fA-F]{2})|'
                    r'(document\.|window\.|eval\()|'
                    r'(base64|\\u[0-9a-fA-F]{4})|'
                    r'(<[^>]*>)'
                )
                self.df['xss_indicator'] = self.df['url'].fillna('').apply(
                    lambda x: int(bool(re.search(xss_patterns, x.lower())))
                )
                
                # Enhanced path traversal detection
                path_patterns = (
                    r'(\.\./|\.\.\\|/\.\.|\\\.\.)|'
                    r'(%2e%2e|%252e)|'
                    r'(\/etc\/|\\windows\\)'
                )
                self.df['path_traversal'] = self.df['url'].fillna('').apply(
                    lambda x: int(bool(re.search(path_patterns, x)))
                )
                
                # Command injection detection
                cmd_patterns = (
                    r'(\||;|`|\$\(|\${)|'
                    r'(cat|echo|wget|curl|ping|nc|netcat)|'
                    r'(\/bin\/|\\system32\\)'
                )
                self.df['cmd_injection'] = self.df['url'].fillna('').apply(
                    lambda x: int(bool(re.search(cmd_patterns, x.lower())))
                )
                
                # Entropy-based features
                def calculate_entropy(string):
                    prob = [float(string.count(c)) / len(string) for c in set(string)]
                    return sum([(p * np.log2(p)) for p in prob]) * -1
                
                self.df['url_entropy'] = self.df['url'].fillna('').apply(calculate_entropy)
                
                # Concatenate all features
                self.df = pd.concat([self.df, url_features_df, count_features_df], axis=1)
                logger.info(f"Extracted {self.df.shape[1]} total features")

        except Exception as e:
            logger.error(f"Error in text feature extraction: {e}")
            raise

    def create_length_features(self):
        """Create features based on length and character counts."""
        try:
            # Add length-based features if relevant columns exist
            text_columns = ['url', 'body', 'headers'] # Adjust based on actual columns
            for col in text_columns:
                if col in self.df.columns:
                    self.df[f'{col}_length'] = self.df[col].fillna('').str.len()
                    self.df[f'{col}_word_count'] = self.df[col].fillna('').str.split().str.len()
            logger.info("Created length-based features")
        except Exception as e:
            logger.error(f"Error in length feature creation: {e}")
            raise

    def scale_numeric_features(self):
        """Scale numeric features using StandardScaler."""
        try:
            numeric_columns = self.df.select_dtypes(include=['int64', 'float64']).columns
            numeric_columns = [col for col in numeric_columns if col != 'classification']
            
            if numeric_columns:
                self.df[numeric_columns] = self.scaler.fit_transform(self.df[numeric_columns])
                logger.info(f"Scaled {len(numeric_columns)} numeric features")
        except Exception as e:
            logger.error(f"Error in feature scaling: {e}")
            raise

    def encode_categorical_features(self):
        """Encode categorical features using LabelEncoder and drop original columns."""
        try:
            # First encode the target variable 'classification'
            if 'classification' in self.df.columns:
                self.df['classification'] = self.label_encoder.fit_transform(self.df['classification'])

            # Then encode all other categorical columns
            categorical_columns = self.df.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                # Create encoded column
                encoded_col = f'{col}_encoded'
                self.df[encoded_col] = self.label_encoder.fit_transform(self.df[col].fillna(''))
                # Drop original column
                self.df.drop(col, axis=1, inplace=True)

            # Verify no object dtypes remain
            remaining_objects = self.df.select_dtypes(include=['object']).columns
            if len(remaining_objects) > 0:
                logger.warning(f"Remaining object columns: {remaining_objects}")
                raise ValueError(f"Found remaining object columns: {remaining_objects}")

            logger.info(f"Encoded {len(categorical_columns)} categorical features and dropped original columns")
        except Exception as e:
            logger.error(f"Error in categorical encoding: {e}")
            raise

    def build_features(self):
        """Execute the complete feature engineering pipeline."""
        logger.info("Starting feature engineering process...")
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Execute feature engineering steps
        self.load_data()
        self.extract_text_features()
        self.create_length_features()
        self.scale_numeric_features()
        self.encode_categorical_features()
        
        # Save feature engineered dataset
        output_file = self.output_dir / 'engineered_features.csv'
        self.df.to_csv(output_file, index=False)
        logger.info(f"Feature engineered dataset saved to {output_file}")
        
        return self.df

def main():
    # Define input and output paths
    input_file = "data/processed/cleaned_dataset.csv"
    output_dir = "data/features"
    
    # Initialize and run the feature builder
    feature_builder = FeatureBuilder(input_file, output_dir)
    try:
        engineered_data = feature_builder.build_features()
        logger.info("Feature engineering completed successfully")
        logger.info(f"Final dataset shape: {engineered_data.shape}")
    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")

if __name__ == "__main__":
    main()