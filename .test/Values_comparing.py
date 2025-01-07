import pandas as pd

# Beispiel-DataFrames
df1 = pd.read_csv('../project_raw_data/stock_data_apple_full_5min.csv')  # Keine header-Option notwendig

df2 = pd.read_csv('../data_preprocessing/data_segmenting/agg_sentiment_VADER.csv')

# Vergleiche, welche Werte in df1['Datetime'] nicht in df2['Segment'] sind
not_in_df1 = df2[~df2['Segment'].isin(df1['Datetime'])]

print(not_in_df1)
