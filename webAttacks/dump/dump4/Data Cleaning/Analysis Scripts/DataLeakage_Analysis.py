import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mutual_info_score
import plotly.io as pio

# Configure plotly to work offline
pio.renderers.default = "browser"

# Load data
df = pd.read_csv("data/csic_database.csv")

def analyze_data_leakage():
    # Store leakage analysis results
    leakage_results = pd.DataFrame(columns=[
        'Feature', 
        'Separation_Score', 
        'Mutual_Information',
        'Risk_Level',
        'Recommendation'
    ])
    
    # Analyze each feature
    for column in df.columns:
        if column != 'classification':
            # Calculate mutual information
            if df[column].dtype in ['int64', 'float64']:
                mi_score = mutual_info_score(df[column], df['classification'])
            else:
                # For categorical features, use label encoding
                mi_score = mutual_info_score(pd.factorize(df[column])[0], df['classification'])
            
            # Calculate separation score (how well classes are separated)
            if df[column].dtype in ['int64', 'float64']:
                class_0_dist = df[df['classification'] == 0][column]
                class_1_dist = df[df['classification'] == 1][column]
                
                if len(class_0_dist) > 0 and len(class_1_dist) > 0:
                    overlap = (min(class_0_dist.max(), class_1_dist.max()) - 
                             max(class_0_dist.min(), class_1_dist.min()))
                    separation_score = 1 - (overlap / (class_0_dist.max() - class_0_dist.min()))
                else:
                    separation_score = 0
            else:
                # For categorical features, check value overlap
                values_class_0 = set(df[df['classification'] == 0][column].unique())
                values_class_1 = set(df[df['classification'] == 1][column].unique())
                separation_score = 1 - len(values_class_0.intersection(values_class_1)) / len(values_class_0.union(values_class_1))
            
            # Determine risk level
            if mi_score > 0.9 or separation_score > 0.95:
                risk_level = 'High'
                recommendation = 'Consider removing or investigating this feature'
            elif mi_score > 0.7 or separation_score > 0.8:
                risk_level = 'Medium'
                recommendation = 'Monitor this feature carefully'
            else:
                risk_level = 'Low'
                recommendation = 'Feature appears safe to use'
            
            # Add to results
            new_row = pd.DataFrame({
                'Feature': [column],
                'Separation_Score': [separation_score],
                'Mutual_Information': [mi_score],
                'Risk_Level': [risk_level],
                'Recommendation': [recommendation]
            })
            leakage_results = pd.concat([leakage_results, new_row], ignore_index=True)
    
    return leakage_results

def visualize_leakage(leakage_results):
    # Create visualization
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Feature Separation Scores', 'Mutual Information Scores'),
        vertical_spacing=0.2
    )
    
    # Sort by separation score
    sorted_by_separation = leakage_results.sort_values('Separation_Score', ascending=True)
    
    # Color mapping for risk levels
    colors = {'High': 'red', 'Medium': 'orange', 'Low': 'green'}
    
    # Add separation score bars
    fig.add_trace(
        go.Bar(
            x=sorted_by_separation['Separation_Score'],
            y=sorted_by_separation['Feature'],
            orientation='h',
            marker_color=[colors[risk] for risk in sorted_by_separation['Risk_Level']],
            name='Separation Score'
        ),
        row=1, col=1
    )
    
    # Sort by mutual information
    sorted_by_mi = leakage_results.sort_values('Mutual_Information', ascending=True)
    
    # Add mutual information bars
    fig.add_trace(
        go.Bar(
            x=sorted_by_mi['Mutual_Information'],
            y=sorted_by_mi['Feature'],
            orientation='h',
            marker_color=[colors[risk] for risk in sorted_by_mi['Risk_Level']],
            name='Mutual Information'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=1200,
        width=1000,
        title_text="Data Leakage Analysis",
        showlegend=False
    )
    
    return fig

def main():
    # Perform leakage analysis
    print("Analyzing data leakage...")
    leakage_results = analyze_data_leakage()
    
    # Create visualization
    print("Creating visualization...")
    fig = visualize_leakage(leakage_results)
    
    # Show results
    print("\nDetailed Leakage Analysis Results:")
    print(leakage_results.sort_values('Risk_Level', ascending=False))
    
    # Show visualization
    fig.show()
    
    # Print high-risk features
    high_risk = leakage_results[leakage_results['Risk_Level'] == 'High']
    if not high_risk.empty:
        print("\nWARNING: The following features show signs of potential data leakage:")
        for _, row in high_risk.iterrows():
            print(f"- {row['Feature']} (Separation Score: {row['Separation_Score']:.3f}, "
                  f"Mutual Information: {row['Mutual_Information']:.3f})")

if __name__ == "__main__":
    main()