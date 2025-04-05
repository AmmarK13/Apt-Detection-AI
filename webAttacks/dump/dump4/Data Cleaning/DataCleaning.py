import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re
from urllib.parse import unquote
import warnings
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

class DataCleaner:
    def __init__(self):
        self.essential_columns = ['content', 'content-type', 'Cookie', 'Length', 'Method', 'URL', 'User-Agent']
        self.header_columns = ['Accept', 'Accept-encoding', 'Accept-charset', 'Cache-Control', 'Pragma']
        self.expected_types = {
            'Method': 'object',
            'URL': 'object',
            'Length': 'int64',
            'content': 'object',
            'content-type': 'object',
            'Cookie': 'object',
            'User-Agent': 'object'
        }

    def load_data(self, filepath):
        print("Loading dataset...")
        return pd.read_csv(filepath)

    def drop_unnamed_columns(self, df):
        print("Checking for unnamed columns...")
        unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
        if unnamed_cols:
            print(f"Dropping unnamed columns: {unnamed_cols}")
            df = df.drop(columns=unnamed_cols)
        return df

    def handle_missing_columns(self, df):
        print("Checking for columns with high missing values...")
        missing_percentages = df.isnull().sum() / len(df) * 100
        columns_to_drop = missing_percentages[missing_percentages >= 50].index
        if len(columns_to_drop) > 0:
            print(f"Dropping columns with >50% missing values: {list(columns_to_drop)}")
            columns_to_drop = [col for col in columns_to_drop if col not in self.essential_columns]
            df = df.drop(columns=columns_to_drop)
        return df, columns_to_drop

    def ensure_essential_columns(self, df):
        print("Ensuring essential columns exist...")
        defaults = {
            'content': '',
            'content-type': 'application/x-www-form-urlencoded',
            'Cookie': '',
            'Length': 0
        }
        for col, default_value in defaults.items():
            if col not in df.columns:
                df[col] = default_value
        return df

    def fill_missing_values(self, df):
        print("Handling missing values...")
        df['content'] = df.apply(lambda x: '' if x['Method'] == 'GET' else 
                               (x['content'] if pd.notna(x['content']) else ''), axis=1)
        df['content-type'] = df.apply(lambda x: '' if x['Method'] == 'GET' else 
                                    (x['content-type'] if pd.notna(x['content-type']) else 
                                    'application/x-www-form-urlencoded'), axis=1)
        df['Cookie'] = df['Cookie'].fillna('')
        df['Length'] = df['Length'].fillna(0)
        return df

    def standardize_values(self, df):
        print("Standardizing values...")
        df['Method'] = df['Method'].str.upper()
        if 'content-type' in df.columns:
            df['content-type'] = df['content-type'].str.lower()
        return df

    @staticmethod
    def clean_url(url):
        if pd.isna(url):
            return ''
        url = unquote(url)
        url = re.sub('/+', '/', url)
        return url.rstrip('/')

    def clean_urls(self, df):
        print("Cleaning URLs...")
        df['URL'] = df['URL'].apply(self.clean_url)
        return df

    def clean_user_agent(self, df):
        print("Cleaning User-Agent strings...")
        df['User-Agent'] = df['User-Agent'].fillna('unknown')
        df['User-Agent'] = df['User-Agent'].str.lower()
        return df

    def standardize_headers(self, df):
        print("Standardizing headers...")
        for col in self.header_columns:
            if col in df.columns:
                df[col] = df[col].fillna('')
                df[col] = df[col].str.lower()
        return df

    def verify_data_types(self, df):
        print("Verifying data types...")
        for col, expected_type in self.expected_types.items():
            if col in df.columns:
                current_type = df[col].dtype
                if str(current_type) != expected_type:
                    print(f"Converting {col} from {current_type} to {expected_type}")
                    if expected_type == 'int64':
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        return df

    def validate_data(self, df):
        print("\nValidating cleaned data...")
        validation_checks = {
            'No empty URLs': df['URL'].str.len().gt(0).all(),
            'Valid HTTP methods': df['Method'].isin(['GET', 'POST', 'PUT', 'DELETE', 'HEAD']).all(),
            'Non-negative lengths': (df['Length'] >= 0).all(),
            'No null values': df.isnull().sum().sum() == 0
        }
        for check, result in validation_checks.items():
            print(f"{check}: {'✓' if result else '✗'}")
        return df

    def generate_report(self, df, original_shape, columns_to_drop):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Create subplot figure with 4 rows, specifying subplot types
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=(
                "Missing Values by Column",
                "Data Types Distribution",
                "Unique Values Distribution",
                "Column Value Counts (Top 10)",
                "Dataset Size Comparison",
                "Classification Distribution",
                "HTTP Methods Distribution",
                "Content Type Distribution"
            ),
            specs=[
                [{"type": "bar"}, {"type": "pie"}],
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "pie"}],
                [{"type": "pie"}, {"type": "pie"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # 1. Missing Values
        missing_values = df.isnull().sum()
        fig.add_trace(
            go.Bar(x=missing_values.index, y=missing_values.values, name="Missing Values"),
            row=1, col=1
        )
        
        # 2. Data Types
        dtype_counts = df.dtypes.value_counts()
        fig.add_trace(
            go.Pie(labels=dtype_counts.index.astype(str), values=dtype_counts.values, name="Data Types"),
            row=1, col=2
        )
        
        # 3. Unique Values
        unique_counts = df.nunique().sort_values(ascending=True)
        fig.add_trace(
            go.Bar(x=unique_counts.index, y=unique_counts.values, name="Unique Values"),
            row=2, col=1
        )
        
        # 4. Top 10 values in selected columns
        selected_col = 'URL'  # You can change this to any column
        value_counts = df[selected_col].value_counts().head(10)
        fig.add_trace(
            go.Bar(x=value_counts.index, y=value_counts.values, name=f"Top 10 {selected_col}"),
            row=2, col=2
        )
        
        # 5. Dataset Size Comparison
        fig.add_trace(
            go.Bar(
                x=['Original', 'Cleaned'],
                y=[original_shape[0], df.shape[0]],
                name="Dataset Size",
                marker_color=['lightblue', 'lightgreen']
            ),
            row=3, col=1
        )
        
        # 6. Classification Distribution
        class_counts = df['classification'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=class_counts.index.astype(str),
                values=class_counts.values,
                name="Classification"
            ),
            row=3, col=2
        )
        
        # 7. HTTP Methods Distribution
        method_counts = df['Method'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=method_counts.index,
                values=method_counts.values,
                name="HTTP Methods"
            ),
            row=4, col=1
        )
        
        # 8. Content Type Distribution
        content_type_counts = df['content-type'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=content_type_counts.index,
                values=content_type_counts.values,
                name="Content Types"
            ),
            row=4, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=2000,
            width=1500,
            showlegend=True,
            title_text="Comprehensive Data Quality Report"
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Columns", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        
        fig.update_xaxes(title_text="Columns", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        
        fig.update_xaxes(title_text="Values", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2)
        
        # Show the plot
        fig.show()
        
        # Print text report as before
        print("\nDetailed Text Report:")
        print("------------------------")
        print("\nMissing Values:")
        print(missing_values)
        print("\nData Types:")
        print(df.dtypes)
        print("\nUnique Values per Column:")
        for col in df.columns:
            print(f"{col}: {df[col].nunique()} unique values")
        print(f"Original shape: {original_shape}")
        print(f"New shape: {df.shape}")
        print(f"Duplicates removed: {original_shape[0] - df.shape[0]}")
        
        if len(columns_to_drop) > 0:
            print(f"Columns dropped: {len(columns_to_drop)}")

    def clean_data(self, input_path="data/csic_database.csv", output_path="data/cleaned_csic_database.csv"):
        # Load data
        df = self.load_data(input_path)
        original_shape = df.shape

        # Clean data
        df = self.drop_unnamed_columns(df)
        df, columns_to_drop = self.handle_missing_columns(df)
        df = df.drop_duplicates()
        df = self.ensure_essential_columns(df)
        df = self.fill_missing_values(df)
        df = self.standardize_values(df)
        df = self.clean_urls(df)
        df = self.clean_user_agent(df)
        df = self.standardize_headers(df)
        
        # Save cleaned dataset
        df.to_csv(output_path, index=False)
        print(f"\nCleaned dataset saved to: {output_path}")
        
        return df

if __name__ == "__main__":
    # Example usage
    cleaner = DataCleaner()
    cleaned_df = cleaner.clean_data()
    
    # Verify the cleaned data
    print("\nVerifying cleaned data...")
    cleaned_df = cleaner.validate_data(cleaned_df)  # Changed from verify_data to validate_data
    
    # Generate comprehensive data quality report
    original_shape = (len(cleaned_df), len(cleaned_df.columns))
    cleaner.generate_report(cleaned_df, original_shape, [])
