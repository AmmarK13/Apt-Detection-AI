import pandas as pd
import numpy as np
from pathlib import Path
import random

def generate_synthetic_data(output_path, num_samples=100):
    """
    Generate synthetic dataset for testing the data transformer
    
    Args:
        output_path (str): Path where the synthetic dataset will be saved
        num_samples (int): Number of samples to generate
        
    Returns:
        pd.DataFrame: Generated synthetic dataset
    """
    
    # Initialize random seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic data
    data = {
        'cookie': [f'sessionid={random.randint(1000, 9999)}; userid={random.randint(100, 999)}' 
                  if random.random() > 0.2 else '' for _ in range(num_samples)],
        
        'url': [f'{"http" if random.random() > 0.5 else "https"}://example.com/{random.choice(["login", "home", "profile", "search"])}?id={random.randint(1, 100)}'
                for _ in range(num_samples)],
        
        'method': [random.choice(['GET', 'POST', 'PUT', 'INVALID_METHOD']) 
                  for _ in range(num_samples)],
        
        'host': [random.choice(['localhost:8080', 'localhost:9090', 'invalid:port']) 
                for _ in range(num_samples)],
        
        'classification': [random.choice([0, 1]) 
                         for _ in range(num_samples)]
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create some edge cases
    if num_samples >= 10:
        # Empty values
        df.loc[0, 'cookie'] = ''
        df.loc[1, 'url'] = ''
        # Invalid URL format
        df.loc[2, 'url'] = 'example.com/no-protocol'
        # Missing protocol
        df.loc[3, 'url'] = '//example.com/missing-protocol'
        # Invalid method
        df.loc[4, 'method'] = 'INVALID'
        # Invalid host
        df.loc[5, 'host'] = 'unknown:1234'
    
    # Save the dataset
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    
    return df

def main():
    """
    Example usage of the synthetic data generator
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic dataset for transformer testing')
    parser.add_argument('--output', type=str, required=True, help='Output path for the synthetic dataset')
    parser.add_argument('--samples', type=int, default=100, help='Number of samples to generate')
    
    args = parser.parse_args()
    
    try:
        df = generate_synthetic_data(args.output, args.samples)
        print(f"Successfully generated {len(df)} samples at: {args.output}")
        print("\nSample data preview:")
        print(df.head())
        
    except Exception as e:
        print(f"Error generating synthetic data: {e}")

if __name__ == "__main__":
    main()