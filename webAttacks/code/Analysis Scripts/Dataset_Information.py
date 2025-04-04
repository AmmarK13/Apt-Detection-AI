import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Get data
df = pd.read_csv("data/csic_database.csv")

# Detailed dataset information
basic_info = pd.DataFrame({
    'Total Rows': [len(df)],
    'Total Columns': [len(df.columns)],
    'Duplicate Rows': [df.duplicated().sum()],
    'Memory Usage (MB)': [round(df.memory_usage().sum() / 1024**2, 2)],
    'Missing Values': [df.isnull().sum().sum()]
})

# Get column information and convert dtypes to strings
column_info = pd.DataFrame({
    'Column Name': df.columns,
    'Data Type': [str(dtype) for dtype in df.dtypes],
    'Non-Null Count': df.count(),
    'Null Count': df.isnull().sum(),
    'Unique Values': [df[col].nunique() for col in df.columns]
})

# Create subplot with three tables
fig = make_subplots(
    rows=3, cols=1,
    subplot_titles=('Basic Dataset Information', 'Column Details', 'Dataset Summary'),
    vertical_spacing=0.1,
    specs=[[{"type": "table"}],
           [{"type": "table"}],
           [{"type": "table"}]]
)

# Add basic info table
fig.add_trace(
    go.Table(
        header=dict(values=list(basic_info.columns),
                   fill_color='paleturquoise',
                   align='left'),
        cells=dict(values=[basic_info[col].astype(str) for col in basic_info.columns],
                  fill_color='lavender',
                  align='left')
    ),
    row=1, col=1
)

# Add column info table
fig.add_trace(
    go.Table(
        header=dict(values=list(column_info.columns),
                   fill_color='paleturquoise',
                   align='left'),
        cells=dict(values=[column_info[col].astype(str) for col in column_info.columns],
                  fill_color='lavender',
                  align='left')
    ),
    row=2, col=1
)

# Create and add console output as a table
console_output = pd.DataFrame({
    'Section': ['Basic Dataset Information', 'Column Information'],
    'Details': [basic_info.to_string(), column_info.to_string()]
})

fig.add_trace(
    go.Table(
        header=dict(values=list(console_output.columns),
                   fill_color='paleturquoise',
                   align='left'),
        cells=dict(values=[console_output[col] for col in console_output.columns],
                  fill_color='lavender',
                  align='left')
    ),
    row=3, col=1
)

# Update layout
fig.update_layout(
    height=1200,  # Increased height to accommodate the new table
    width=1000,
    showlegend=False,
    title_text="Comprehensive Dataset Information"
)

# Display all information in browser
fig.show()
