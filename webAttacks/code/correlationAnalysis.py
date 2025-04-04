import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pointbiserialr
from scipy.stats import chi2_contingency

class CorrelationAnalyzer:
    def __init__(self):
        self.correlation_matrix = None
        self.feature_variance = None
        
    def cramers_v(self, x, y):
        """Calculate Cramér's V correlation between two categorical variables"""
        try:
            # Check for constant arrays
            if len(x.unique()) == 1 or len(y.unique()) == 1:
                return 0.0
                
            confusion_matrix = pd.crosstab(x, y)
            chi2 = chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum().sum()
            min_dim = min(confusion_matrix.shape) - 1
            
            # Handle division by zero
            if n * min_dim == 0:
                return 0.0
                
            return np.sqrt(chi2 / (n * min_dim))
        except (ValueError, MemoryError):
            return 0.0
    
    def preprocess_dataframe(self, df):
        """Preprocess dataframe to handle different data types"""
        df_processed = df.copy()
        
        # Convert boolean columns to int
        bool_cols = df_processed.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            df_processed[col] = df_processed[col].astype(int)
        
        # Remove string/object columns that can't be correlated
        exclude_cols = ['URL', 'User-Agent', 'content', 'content-type']
        for col in exclude_cols:
            if col in df_processed.columns:
                df_processed = df_processed.drop(col, axis=1)
        
        # Ensure classification is numeric
        if 'classification' in df_processed.columns:
            df_processed['classification'] = pd.factorize(df_processed['classification'])[0]
        
        return df_processed
    
    def calculate_classification_correlations(self, df):
        """Calculate correlations between all features and classification"""
        # Preprocess the dataframe
        df = self.preprocess_dataframe(df)
        
        # Separate features by type
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        correlations = {}
        
        # Calculate correlations for each feature type
        for feature in df.columns:
            if feature != 'classification':
                try:
                    if feature in numerical_cols:
                        # Numerical features: Spearman correlation
                        corr = df[feature].corr(df['classification'], method='spearman')
                    elif feature in categorical_cols:
                        # Categorical features: Cramér's V
                        corr = self.cramers_v(df[feature], df['classification'])
                    else:
                        # Mixed/Binary features: Point Biserial
                        cat_encoded = pd.factorize(df[feature])[0]
                        corr = pointbiserialr(cat_encoded, df['classification'])[0]
                    
                    correlations[feature] = abs(corr) if not pd.isna(corr) else 0.0
                    
                except Exception as e:
                    print(f"Warning: Error calculating correlation for {feature}: {str(e)}")
                    correlations[feature] = 0.0
        
        # Sort by absolute correlation value
        sorted_correlations = pd.Series(correlations).sort_values(ascending=False)
        return sorted_correlations
    
    def plot_classification_correlations(self, correlations):
        """Plot correlations with classification in multiple formats"""
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Bar plot
        correlations.plot(kind='bar', ax=ax1)
        ax1.set_title('Feature Correlations with Attack Classification')
        ax1.set_xlabel('Features')
        ax1.set_ylabel('Absolute Correlation Coefficient')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Heatmap
        heatmap_data = pd.DataFrame({
            'Classification': correlations
        })
        sns.heatmap(heatmap_data.T, annot=True, cmap='RdYlBu_r', center=0, fmt='.2f', ax=ax2)
        ax2.set_title('Correlation Heatmap with Classification')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()
        
        # Additional feature type distribution plot
        plt.figure(figsize=(10, 6))
        correlations.plot(kind='kde')
        plt.title('Distribution of Feature Correlations')
        plt.xlabel('Correlation Coefficient')
        plt.ylabel('Density')
        plt.grid(True)
        plt.show()
    
    def calculate_correlations(self, df, method='mixed'):
        """Calculate correlation matrix using appropriate methods for different feature types"""
        # Preprocess the dataframe
        df = self.preprocess_dataframe(df)
        
        # Separate numerical and categorical columns
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Initialize correlation matrix
        all_cols = list(numerical_cols) + list(categorical_cols)
        n_cols = len(all_cols)
        corr_matrix = pd.DataFrame(np.zeros((n_cols, n_cols)), columns=all_cols, index=all_cols)
        
        # Process in smaller chunks to save memory
        chunk_size = 1000
        
        for i, col1 in enumerate(all_cols):
            for j in range(i):
                col2 = all_cols[j]
                try:
                    if col1 in numerical_cols and col2 in numerical_cols:
                        # Numerical-Numerical: Spearman correlation
                        corr = df[col1].corr(df[col2], method='spearman')
                    elif col1 in categorical_cols and col2 in categorical_cols:
                        # Categorical-Categorical: Cramér's V
                        corr = self.cramers_v(df[col1], df[col2])
                    else:
                        # Numerical-Categorical: Point Biserial correlation
                        if col1 in numerical_cols:
                            num_col, cat_col = col1, col2
                        else:
                            num_col, cat_col = col2, col1
                        
                        # Handle constant arrays
                        if len(df[cat_col].unique()) == 1:
                            corr = 0.0
                        else:
                            cat_encoded = pd.factorize(df[cat_col])[0]
                            try:
                                corr = pointbiserialr(df[num_col], cat_encoded)[0]
                            except (ValueError, MemoryError):
                                corr = 0.0
                    
                    if pd.isna(corr):
                        corr = 0.0
                        
                    corr_matrix.loc[col1, col2] = corr
                    corr_matrix.loc[col2, col1] = corr
                    
                except Exception as e:
                    print(f"Warning: Error calculating correlation between {col1} and {col2}: {str(e)}")
                    corr_matrix.loc[col1, col2] = 0.0
                    corr_matrix.loc[col2, col1] = 0.0
            
            # Diagonal values
            corr_matrix.loc[col1, col1] = 1.0
        
        self.correlation_matrix = corr_matrix
        return self.correlation_matrix
    
    def identify_high_correlations(self, threshold=0.8):
        """Identify highly correlated feature pairs"""
        high_corr = []
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i):
                if abs(self.correlation_matrix.iloc[i, j]) > threshold:
                    col1 = self.correlation_matrix.columns[i]
                    col2 = self.correlation_matrix.columns[j]
                    corr = self.correlation_matrix.iloc[i, j]
                    high_corr.append((col1, col2, corr))
        return high_corr
    
    def calculate_feature_variance(self, df):
        """Calculate variance for each feature"""
        self.feature_variance = df.var()
        return self.feature_variance
    
    def identify_low_variance(self, threshold=0.01):
        """Identify features with low variance"""
        return self.feature_variance[self.feature_variance < threshold]
    
    def plot_correlation_heatmap(self):
        """Plot correlation heatmap"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            self.correlation_matrix,
            annot=True,
            cmap='coolwarm',
            center=0,
            fmt='.2f'
        )
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
    
    def analyze_attack_correlations(self, df):
        """Analyze correlations with attack patterns"""
        attack_corr = df.corr()['classification'].sort_values(ascending=False)
        
        plt.figure(figsize=(12, 6))
        attack_corr.plot(kind='bar')
        plt.title('Feature Correlations with Attack Classification')
        plt.xlabel('Features')
        plt.ylabel('Correlation Coefficient')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
        return attack_corr
    
    def suggest_features_to_remove(self, df, correlation_threshold=0.8, variance_threshold=0.01):
        """Suggest features that might be candidates for removal"""
        suggestions = {
            'high_correlation': [],
            'low_variance': [],
            'keep_security': []
        }
        
        # High correlation analysis
        high_corr = self.identify_high_correlations(correlation_threshold)
        for col1, col2, corr in high_corr:
            suggestions['high_correlation'].append(f"{col1} & {col2} (correlation: {corr:.2f})")
        
        # Low variance analysis
        low_var = self.identify_low_variance(variance_threshold)
        suggestions['low_variance'] = low_var.index.tolist()
        
        # Security-significant features to keep despite correlations
        security_features = [
            'contains_sql_injection',
            'contains_xss',
            'contains_path_traversal',
            'content_special_chars'
        ]
        suggestions['keep_security'] = [f for f in security_features if f in df.columns]
        
        return suggestions

if __name__ == "__main__":
    # Load scaled data
    df = pd.read_csv("data/robust_scaled_csic_database.csv")
    
    # Initialize analyzer
    analyzer = CorrelationAnalyzer()
    
    # Calculate correlations with classification
    classification_correlations = analyzer.calculate_classification_correlations(df)
    
    # Plot correlations
    analyzer.plot_classification_correlations(classification_correlations)
    
    # Print top correlations
    print("\nTop correlations with attack classification:")
    print(classification_correlations.head(10))
    print("\nBottom correlations with attack classification:")
    print(classification_correlations.tail(10))