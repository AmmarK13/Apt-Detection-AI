import pandas as pd
from correlationAnalysis import CorrelationAnalyzer

class FeatureRemovalAnalyzer:
    def __init__(self):
        self.analyzer = CorrelationAnalyzer()
        
    def analyze_features(self, df, correlation_threshold=0.9, variance_threshold=0.001):
        """Analyze features and provide removal recommendations"""
        # Initialize recommendations dictionary
        recommendations = {
            'remove': [],
            'keep': [],
            'consider_removing': []
        }
        
        # Preprocess the dataframe first
        df_processed = self.analyzer.preprocess_dataframe(df)
        
        # Remove constant columns first (both numerical and categorical)
        constant_cols = [col for col in df_processed.columns if df_processed[col].nunique() == 1]
        df_processed = df_processed.drop(constant_cols, axis=1)
        
        # Add constant columns to consider removing
        for col in constant_cols:
            if col != 'classification':
                recommendations['consider_removing'].append({
                    'feature': col,
                    'reason': 'Constant value across all samples',
                    'impact': 'No impact on classification (constant)'
                })
        
        # Separate features and target
        features = df_processed.drop('classification', axis=1)
        target = df_processed['classification']
        
        # Calculate metrics only for non-constant columns
        classification_corr = self.analyzer.calculate_classification_correlations(df_processed)
        self.analyzer.calculate_correlations(features)
        
        # Calculate variance only for numerical columns
        numerical_cols = features.select_dtypes(include=['int64', 'float64']).columns
        df_numerical = features[numerical_cols]
        self.analyzer.calculate_feature_variance(df_numerical)
        
        # Get additional recommendations
        additional_recommendations = self._get_removal_recommendations(
            features, classification_corr, correlation_threshold, variance_threshold
        )
        
        # Merge recommendations
        for key in recommendations:
            recommendations[key].extend(additional_recommendations[key])
        
        return recommendations
    
    def _get_removal_recommendations(self, df, classification_corr, correlation_threshold, variance_threshold):
        """Get detailed feature removal recommendations"""
        recommendations = {
            'remove': [],
            'keep': [],
            'consider_removing': []
        }
        
        # Get highly correlated features but only if they have low importance
        high_corr_pairs = self.analyzer.identify_high_correlations(correlation_threshold)
        for col1, col2, corr in high_corr_pairs:
            corr1_with_class = abs(classification_corr.get(col1, 0))
            corr2_with_class = abs(classification_corr.get(col2, 0))
            
            # Only recommend removal if correlation with target is very low
            if min(corr1_with_class, corr2_with_class) < 0.1:
                weaker_feature = col2 if corr1_with_class > corr2_with_class else col1
                stronger_feature = col1 if corr1_with_class > corr2_with_class else col2
                
                recommendations['remove'].append({
                    'feature': weaker_feature,
                    'reason': f'Highly correlated with {stronger_feature} (corr: {corr:.2f})',
                    'impact': f'Classification correlation: {min(corr1_with_class, corr2_with_class):.2f}'
                })
        
        # Get low variance features but only if they're not important for classification
        low_var = self.analyzer.identify_low_variance(variance_threshold)
        for feature in low_var.index:
            if abs(classification_corr.get(feature, 0)) < 0.05:  # Very low importance threshold
                recommendations['consider_removing'].append({
                    'feature': feature,
                    'reason': f'Low variance: {self.analyzer.feature_variance[feature]:.4f}',
                    'impact': f'Classification correlation: {abs(classification_corr.get(feature, 0)):.2f}'
                })
        
        # Important security features to keep
        security_features = [
            'contains_sql_injection',
            'contains_xss',
            'contains_path_traversal',
            'content_special_chars',
            'length',
            'special_char_ratio',
            'entropy'
        ]
        
        for feature in security_features:
            if feature in df.columns:
                recommendations['keep'].append({
                    'feature': feature,
                    'reason': 'Security-critical feature',
                    'impact': f'Classification correlation: {abs(classification_corr.get(feature, 0)):.2f}'
                })
        
        return recommendations
    
    def _visualize_analysis(self, classification_corr):
        """Visualize the feature analysis results"""
        # self.analyzer.plot_classification_correlations(classification_corr)
        # self.analyzer.plot_correlation_heatmap()

def main():
    import os
    
    # Create results directory in the current working directory
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data using relative path
    df = pd.read_csv("./data/robust_scaled_csic_database.csv")
    
    # Initialize analyzer
    remover = FeatureRemovalAnalyzer()
    
    # Get recommendations
    recommendations = remover.analyze_features(df)
    
    # Create a list to store results
    results = []
    
    # Collect recommendations
    for item in recommendations['remove']:
        results.append({
            'feature': item['feature'],
            'status': 'Remove',
            'reason': item['reason'],
            'impact': item['impact']
        })
    
    for item in recommendations['keep']:
        results.append({
            'feature': item['feature'],
            'status': 'Keep',
            'reason': item['reason'],
            'impact': item['impact']
        })
    
    for item in recommendations['consider_removing']:
        results.append({
            'feature': item['feature'],
            'status': 'Consider Removing',
            'reason': item['reason'],
            'impact': item['impact']
        })
    
    # Convert results to DataFrame and save to CSV using relative path
    results_df = pd.DataFrame(results)
    output_path = os.path.join(results_dir, "data/feature_removal_recommendations.csv")
    try:
        results_df.to_csv(output_path, index=False)
        print(f"\nResults have been saved to '{os.path.abspath(output_path)}'")
    except PermissionError:
        print(f"\nWarning: Could not save to {output_path} due to permission denied.")
        # Try saving to current directory instead
        output_path = "feature_removal_recommendations.csv"
        results_df.to_csv(output_path, index=False)
        print(f"Results have been saved to '{os.path.abspath(output_path)}' instead.")
    
    # Print recommendations
    print("\nFeature Removal Recommendations:")
    print("\n1. Features to Remove:")
    for item in recommendations['remove']:
        print(f"  - {item['feature']}")
        print(f"    Reason: {item['reason']}")
        print(f"    Impact: {item['impact']}")
    
    print("\n2. Features to Keep:")
    for item in recommendations['keep']:
        print(f"  - {item['feature']}")
        print(f"    Reason: {item['reason']}")
        print(f"    Impact: {item['impact']}")
    
    print("\n3. Features to Consider Removing:")
    for item in recommendations['consider_removing']:
        print(f"  - {item['feature']}")
        print(f"    Reason: {item['reason']}")
        print(f"    Impact: {item['impact']}")
    
    print(f"\nResults have been saved to '{output_path}'")

if __name__ == "__main__":
    main()