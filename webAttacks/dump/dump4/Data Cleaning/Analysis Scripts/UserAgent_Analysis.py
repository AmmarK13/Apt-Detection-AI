import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Get data
df = pd.read_csv("data/csic_database.csv")

# User-Agent analysis
top_agents = df['User-Agent'].value_counts().head(10)

fig = px.bar(x=top_agents.values,
             y=top_agents.index,
             title='Top 10 Most Common User-Agents',
             orientation='h')

fig.update_layout(
    plot_bgcolor='white',
    xaxis_title='Number of Requests',
    yaxis_title='User-Agent',
    height=500,
    yaxis={'categoryorder':'total ascending'}
)

fig.show()