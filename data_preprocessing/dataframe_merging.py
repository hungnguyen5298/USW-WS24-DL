import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

fundus_df = pd.read_csv(os.path.join('project_raw_data', 'filtered_news_fundus.csv'))
reddit_df = pd.read_csv(os.path.join('project_raw_data', 'filtered_reddit_praw.csv'))
newsapi_df = pd.read_csv(os.path.join('project_raw_data', 'filtered_news_newsapi.csv'))
stock_df = pd.read_csv(os.path.join('project_raw_data', 'filtered_stock.csv'))

print(fundus_df.head())
print(reddit_df.head())
print(newsapi_df.head())
print(stock_df.head())

fundus_df_selected = fundus_df[["", ]]
reddit_df_selected = reddit_df[["", ]]
newsapi_df_selected = newsapi_df[["", ]]
stock_df_selected = stock_df[["", ]]

