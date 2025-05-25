import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Get data and check column names
df = pd.read_csv("data/csic_database.csv")

# Create a subplot with multiple metrics
fig = make_subplots(rows=2, cols=2,
                    subplot_titles=('Cache Control Usage',
                                  'Accept Types Distribution',
                                  'Connection Types',
                                  'Method Distribution'))

# Cache Control
cache_by_class = df.groupby(['classification', 'Cache-Control']).size().reset_index(name='count')
fig.add_trace(
    go.Bar(x=cache_by_class['classification'],
           y=cache_by_class['count'],
           name='Cache Control'),
    row=1, col=1
)

# Accept Types
accept_by_class = df.groupby(['classification', 'Accept']).size().reset_index(name='count')
fig.add_trace(
    go.Bar(x=accept_by_class['classification'],
           y=accept_by_class['count'],
           name='Accept Types'),
    row=1, col=2
)

# Connection Types
conn_by_class = df.groupby(['classification', 'connection']).size().reset_index(name='count')
fig.add_trace(
    go.Bar(x=conn_by_class['classification'],
           y=conn_by_class['count'],
           name='Connection'),
    row=2, col=1
)

# Method Distribution (fixed capitalization)
method_by_class = df.groupby(['classification', 'Method']).size().reset_index(name='count')
fig.add_trace(
    go.Bar(x=method_by_class['classification'],
           y=method_by_class['count'],
           name='Method'),
    row=2, col=2
)

fig.update_layout(height=800, title_text="Request Characteristics by Classification")
fig.show()