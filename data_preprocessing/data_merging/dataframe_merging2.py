import pandas as pd

# Merging function defining
def merge_dataframes(df1, df2, key1, key2, join_type='inner'):
    df1[key1] = pd.to_datetime(df1[key1])
    df2[key2] = pd.to_datetime(df2[key2])

    merged_df = pd.merge(df1, df2, left_on=key1, right_on=key2, how=join_type)
    return merged_df

# Data importing
vader = pd.read_csv('../data_time_shifting/vader_shifted.csv')
finbert = pd.read_csv('../data_time_shifting/finbert_shifted.csv')
stock = pd.read_csv('../stock_data_feature_engineering/stock_data_apple_indicators.csv')

'''vader = vader[14:]
finbert = finbert[14:]'''

# DataFrame for VADER creating
result = merge_dataframes(vader, stock, 'Timestamp', 'Timestamp', 'inner')
result = result.drop(columns=['timestamp_shifted', 'Volume'])
result = result[['Timestamp', 'VADER_Positive', 'VADER_Neutral', 'VADER_Negative', 'Open', 'High', 'Low', 'Volatility', 'RSI', 'OBV', 'ATR', 'Profit_Trend_Label']]
result.to_csv('vader_stock_joined.csv', index=False)

# DataFrame for FinBERT creating
result = merge_dataframes(finbert, stock, 'Timestamp', 'Timestamp', 'inner')
result = result.drop(columns=['timestamp_shifted', 'Volume'])
result = result[['Timestamp', 'Positive_Prob', 'Neutral_Prob', 'Negative_Prob', 'Open', 'High', 'Low', 'Volatility', 'RSI', 'OBV', 'ATR', 'Profit_Trend_Label']]
result.to_csv('finbert_stock_joined.csv', index=False)
