import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Get data
df = pd.read_csv("data/csic_database.csv")

# HTTP Methods analysis
fig = px.histogram(df,
                  x='Method',
                  color='classification',
                  title='HTTP Methods by Classification',
                  barmode='group',
                  color_discrete_sequence=['#2ecc71', '#e74c3c'])

fig.update_layout(
    plot_bgcolor='white',
    xaxis_title='HTTP Method',
    yaxis_title='Count',
    bargap=0.2
)

fig.show()