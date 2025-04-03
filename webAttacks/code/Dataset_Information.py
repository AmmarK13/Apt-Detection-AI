import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Get data
df = pd.read_csv("data/csic_database.csv")

# Basic dataset information
basic_info = pd.DataFrame({
    'Total Rows': [len(df)],
    'Total Columns': [len(df.columns)],
    'Duplicate Rows': [df.duplicated().sum()],
})

basic_info