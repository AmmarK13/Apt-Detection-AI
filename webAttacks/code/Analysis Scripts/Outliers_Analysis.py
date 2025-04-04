import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Configure plotly to work offline
pio.renderers.default = "browser"

# Get data
df = pd.read_csv("data/csic_database.csv")

# 1. Analyze categorical columns for unusual patterns and outliers
def analyze_categorical_outliers(df, column):
    # Get value counts and percentages
    value_counts = df[column].value_counts()
    value_percentages = (value_counts / len(df)) * 100
    
    # Find rare categories (less than 1% occurrence) - these are potential outliers
    rare_categories = value_percentages[value_percentages < 1]
    
    # Check distribution across target class
    class_distribution = pd.crosstab(df[column], df['classification'], normalize='index') * 100
    
    # Identify suspicious patterns - these are outliers where distribution is heavily skewed
    suspicious = class_distribution[(class_distribution == 0).any(axis=1) | (class_distribution == 100).any(axis=1)]
    
    # Calculate statistical outliers for numerical columns
    outliers = {}
    if df[column].dtype in ['int64', 'float64']:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        outlier_mask = (df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))
        outliers = {
            'lower_bound': Q1 - 1.5 * IQR,
            'upper_bound': Q3 + 1.5 * IQR,
            'outlier_count': outlier_mask.sum()
        }
    
    return {
        'column': column,
        'unique_values': len(value_counts),
        'rare_categories': len(rare_categories),
        'rare_items': rare_categories,
        'suspicious_patterns': len(suspicious),
        'value_counts': value_counts,
        'class_distribution': class_distribution,
        'statistical_outliers': outliers
    }

# Separate columns by type
categorical_cols = df.select_dtypes(include=['object']).columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Create main figure with all subplots
main_fig = make_subplots(
    rows=5, cols=2,
    subplot_titles=tuple([f"Analysis for {col}" for col in categorical_cols if col != 'classification'] + 
                        ['URL Length', 'URL Special Characters', 'Content Distribution']),
    vertical_spacing=0.08,
    horizontal_spacing=0.1
)

current_row = 1
current_col = 1
max_row = 5

# 2. Analyze and plot outliers for categorical columns
for col in categorical_cols:
    if col != 'classification' and current_row <= max_row:
        analysis = analyze_categorical_outliers(df, col)
        
        # Plot value distribution highlighting outliers (rare categories)
        value_counts = analysis['value_counts'].head(10)
        colors = ['red' if x in analysis['rare_items'].index else '#FFFACD' for x in value_counts.index]  # Light yellow (Lemon Chiffon) color
        
        main_fig.add_trace(
            go.Bar(x=value_counts.index,
                  y=value_counts.values,
                  name=f'{col} Count',
                  marker_color=colors),
            row=current_row, col=current_col
        )
        
        if current_col == 2:
            current_row += 1
            current_col = 1
        else:
            current_col = 2

if current_col == 2:
    current_row += 1
    current_col = 1

# Add URL analysis plots with outlier detection
if current_row <= max_row:
    url_length_analysis = analyze_categorical_outliers(df, 'url_length')
    main_fig.add_trace(
        go.Box(x=df['classification'], 
               y=df['url_length'],
               name='URL Length',
               boxpoints='outliers'),  # Show outlier points
        row=current_row, col=current_col
    )

    if current_col == 2:
        current_row += 1
        current_col = 1
    else:
        current_col = 2

if current_row <= max_row:
    special_chars_analysis = analyze_categorical_outliers(df, 'url_special_chars')
    main_fig.add_trace(
        go.Box(x=df['classification'], 
               y=df['url_special_chars'],
               name='Special Chars',
               boxpoints='outliers'),  # Show outlier points
        row=current_row, col=current_col
    )

    if current_col == 2:
        current_row += 1
        current_col = 1
    else:
        current_col = 2

# Add content pattern analysis with outlier highlighting
if current_row <= max_row:
    content_patterns = pd.crosstab([df['content-type'], df['Method']], 
                                  df['classification'],
                                  normalize='index') * 100
    
    # Identify outlier patterns (extreme distributions)
    outlier_mask = (content_patterns < content_patterns.quantile(0.05)) | (content_patterns > content_patterns.quantile(0.95))
    colors = ['red' if any(outlier_mask.loc[idx]) else 'green' for idx in content_patterns.index]

    main_fig.add_trace(
        go.Bar(x=content_patterns.reset_index()['content-type'],
               y=content_patterns[content_patterns.columns[0]],
               name='Content Distribution',
               marker_color=colors),
        row=current_row, col=current_col
    )

# Update layout
main_fig.update_layout(
    height=1200,
    width=1200,
    title_text="Complete Analysis Dashboard with Outliers Highlighted",
    showlegend=True
)

# Show the combined figure
main_fig.show()

# Create summary of unusual patterns and outliers
unusual_patterns = pd.DataFrame(columns=['Pattern_Type', 'Description', 'Occurrence_Rate', 'Risk_Level', 'Outlier_Type'])

# Add patterns that might indicate attacks
for col in categorical_cols:
    if col != 'classification':
        analysis = analyze_categorical_outliers(df, col)
        class_dist = analysis['class_distribution']
        
        # Add statistical outliers
        if analysis['statistical_outliers']:
            outliers = analysis['statistical_outliers']
            new_row = pd.DataFrame({
                'Pattern_Type': [col],
                'Description': [f"Statistical outliers detected: {outliers['outlier_count']} values outside [{outliers['lower_bound']:.2f}, {outliers['upper_bound']:.2f}]"],
                'Occurrence_Rate': [f"{outliers['outlier_count'] / len(df):.2%}"],
                'Risk_Level': ['High'],
                'Outlier_Type': ['Statistical']
            })
            unusual_patterns = pd.concat([unusual_patterns, new_row], ignore_index=True)
        
        # Add distribution-based outliers
        suspicious = class_dist[(class_dist == 0).any(axis=1) | (class_dist == 100).any(axis=1)]
        for idx in suspicious.index:
            class_values = class_dist.loc[idx]
            majority_class = class_values.idxmax()
            
            new_row = pd.DataFrame({
                'Pattern_Type': [col],
                'Description': [f"Value '{idx}' appears predominantly in {majority_class} requests"],
                'Occurrence_Rate': [f"{len(df[df[col] == idx]) / len(df):.2%}"],
                'Risk_Level': ['High' if class_values.min() == 0 else 'Medium'],
                'Outlier_Type': ['Distribution']
            })
            unusual_patterns = pd.concat([unusual_patterns, new_row], ignore_index=True)

unusual_patterns