import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pickle

class FeatureScaler:
    def __init__(self):
        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.robust_scaler = RobustScaler(quantile_range=(25.0, 75.0))
        self.numerical_columns = [
            'url_length', 
            'path_depth', 
            'query_length', 
            'param_count',
            'content_length', 
            'content_special_chars', 
            'content_numbers'
        ]
        
    def fit_transform_data(self, df, scaler_type='robust'):
        """Fit and transform the data using specified scaler"""
        df_scaled = df.copy()
        
        if scaler_type == 'standard':
            scaler = self.standard_scaler
        elif scaler_type == 'minmax':
            scaler = self.minmax_scaler
        else:
            scaler = self.robust_scaler
            
        # Scale only numerical columns, keep other columns unchanged
        df_scaled[self.numerical_columns] = scaler.fit_transform(df[self.numerical_columns])
        
        # Ensure all original columns are preserved
        for col in df.columns:
            if col not in self.numerical_columns:
                df_scaled[col] = df[col]
        
        return df_scaled
    
    def transform_data(self, df, scaler_type='robust'):
        """Transform new data using fitted scaler"""
        df_scaled = df.copy()
        
        if scaler_type == 'standard':
            scaler = self.standard_scaler
        elif scaler_type == 'minmax':
            scaler = self.minmax_scaler
        else:
            scaler = self.robust_scaler
            
        df_scaled[self.numerical_columns] = scaler.transform(df[self.numerical_columns])
        
        return df_scaled
    
    def save_scalers(self, path):
        """Save fitted scalers for later use"""
        with open(f'{path}/standard_scaler.pkl', 'wb') as f:
            pickle.dump(self.standard_scaler, f)
        with open(f'{path}/minmax_scaler.pkl', 'wb') as f:
            pickle.dump(self.minmax_scaler, f)
        with open(f'{path}/robust_scaler.pkl', 'wb') as f:
            pickle.dump(self.robust_scaler, f)
    
    def load_scalers(self, path):
        """Load saved scalers"""
        with open(f'{path}/standard_scaler.pkl', 'rb') as f:
            self.standard_scaler = pickle.load(f)
        with open(f'{path}/minmax_scaler.pkl', 'rb') as f:
            self.minmax_scaler = pickle.load(f)
        with open(f'{path}/robust_scaler.pkl', 'rb') as f:
            self.robust_scaler = pickle.load(f)
    
    def analyze_scaling_results(self, df_original, df_scaled):
        """Analyze and visualize scaling results"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        for column in self.numerical_columns:
            plt.figure(figsize=(12, 4))
            
            # Original vs Scaled distribution
            plt.subplot(1, 2, 1)
            sns.boxplot(data=df_original[column])
            plt.title(f'Original {column}')
            
            plt.subplot(1, 2, 2)
            sns.boxplot(data=df_scaled[column])
            plt.title(f'Scaled {column}')
            
            plt.tight_layout()
            plt.show()
            
            # Print statistics
            print(f"\nFeature: {column}")
            print("Original Stats:")
            print(df_original[column].describe())
            print("\nScaled Stats:")
            print(df_scaled[column].describe())

if __name__ == "__main__":
    # Load featured data
    df = pd.read_csv("data/featured_csic_database.csv")
    
    # Initialize scaler
    scaler = FeatureScaler()
    
    # Scale features using RobustScaler
    df_robust = scaler.fit_transform_data(df, scaler_type='robust')
    
    # Save scaled dataset
    df_robust.to_csv("data/robust_scaled_csic_database.csv", index=False)
    
    # Save scalers for future use
    scaler.save_scalers("data")
    
    print("Robust scaling completed and saved.")
    print(f"Robust scaled shape: {df_robust.shape}")
    
    # Analyze scaling results
    scaler.analyze_scaling_results(df, df_robust)