import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Get data
df = pd.read_csv("data/csic_database.csv")

# Target variable distribution
fig = px.pie(df, 
             names='classification',
             title='Distribution of Normal vs Anomalous Requests',
             color_discrete_sequence=['#2ecc71', '#e74c3c'])

fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(plot_bgcolor='white')

fig.show()