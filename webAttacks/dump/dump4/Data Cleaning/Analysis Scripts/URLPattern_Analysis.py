import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Get data
df = pd.read_csv("data/csic_database.csv")

# Calculate URL length
df['url_length'] = df['URL'].str.len()

# URL analysis with histogram
fig = px.histogram(df,
                  x='url_length',
                  color='classification',
                  title='URL Length Distribution by Classification',
                  marginal='box',  # Adds box plot on the margin
                  opacity=0.7,  # Makes bars semi-transparent
                  labels={'url_length': 'URL Length',
                         'classification': 'Request Type',
                         'count': 'Number of Requests'},
                  color_discrete_sequence=['#2ecc71', '#e74c3c'])

fig.update_layout(
    plot_bgcolor='white',
    barmode='overlay',  # Overlays the histograms
    xaxis_title='URL Length',
    yaxis_title='Number of Requests'
)

fig.show()