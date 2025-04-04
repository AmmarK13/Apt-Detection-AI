import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Get data
try:
    df = pd.read_csv("data/csic_database.csv")
except FileNotFoundError:
    raise FileNotFoundError("The data file 'csic_database.csv' was not found in the data directory")

# Verify required columns exist
required_columns = ['URL', 'User-Agent']
if not all(col in df.columns for col in required_columns):
    raise ValueError(f"Missing required columns. Dataset must contain: {required_columns}")

# Create numerical features
df['url_length'] = df['URL'].str.len()
df['user_agent_length'] = df['User-Agent'].str.len()

# Create subplots: box plot and histogram for each feature
numerical_cols = ['url_length', 'user_agent_length']
fig = make_subplots(
    rows=len(numerical_cols), 
    cols=2,
    subplot_titles=[
        'URL Length Distribution', 'URL Length Box Plot',
        'User-Agent Length Distribution', 'User-Agent Length Box Plot'
    ]
)

# Add traces for each feature
for idx, col in enumerate(numerical_cols, 1):
    # Add histogram
    fig.add_trace(
        go.Histogram(x=df[col], name=f'{col} distribution', nbinsx=50),
        row=idx, col=1
    )
    
    # Add box plot
    fig.add_trace(
        go.Box(x=df[col], name=col, boxpoints='outliers', boxmean=True),
        row=idx, col=2
    )

# Update layout
fig.update_layout(
    height=800,
    title_text="Distribution and Numerical Outlier Analysis",
    showlegend=False,
    template='plotly_white'
)

# Update x-axes titles
for i in range(1, 3):
    fig.update_xaxes(title_text="Length", row=1, col=i)
    fig.update_xaxes(title_text="Length", row=2, col=i)

# Update y-axes titles
for i in range(1, 3):
    fig.update_yaxes(title_text="Count" if i == 1 else "", row=1, col=i)
    fig.update_yaxes(title_text="Count" if i == 1 else "", row=2, col=i)

fig.show()