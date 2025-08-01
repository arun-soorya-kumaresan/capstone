# data.py
import pandas as pd

# --- Load Data ---
df = pd.read_csv("dc_sustainability_data.csv", index_col=0)
# --- This fixes the 'AttributeError' in the Scenario Planning tab ---
df.index = pd.to_datetime(df.index)