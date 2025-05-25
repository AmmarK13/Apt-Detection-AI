import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Get data
df = pd.read_csv("data/csic_database.csv")

# Missing values analysis
missing_data = pd.DataFrame({
    'Column': df.columns,
    'Missing Values': df.isnull().sum(),
    'Missing Percentage': (df.isnull().sum() / len(df)) * 100
}).sort_values('Missing Percentage', ascending=True)

# Create horizontal bar chart for missing values
fig = px.bar(missing_data,
             y='Column',
             x='Missing Percentage',
             title='Missing Values in Each Column',
             orientation='h')

fig.update_layout(
    plot_bgcolor='white',
    xaxis_title='Percentage of Missing Values',
    yaxis_title='Column Name',
    height=600
)

fig.show()