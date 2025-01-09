import pandas as pd

# Daten importieren
stock = pd.read_csv('../stock_data_feature_engineering/stock_data_apple_indicators.csv')
vader = pd.read_csv('../data_merging/vader_stock_joined.csv')
finbert = pd.read_csv('../data_merging/finber_stock_joined.csv')

# nach