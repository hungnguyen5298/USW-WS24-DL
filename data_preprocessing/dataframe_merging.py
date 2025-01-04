import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

fundus_df = pd.read_csv('../project_raw_data/filtered_news_fundus.csv')
reddit_df = pd.read_csv('../project_raw_data/filtered_reddit_praw.csv')
newsapi_df = pd.read_csv('../project_raw_data/filtered_full_news_newsapi_org.csv')
alpha_df = pd.read_csv('../project_raw_data/filtered_news_alphavantage.csv')
google_df = pd.read_csv('../project_raw_data/rss_google_news.csv')
yahoo_df = pd.read_csv('../project_raw_data/rss_yahoofinance.csv')

print(fundus_df.columns)
print(reddit_df.columns)
print(newsapi_df.columns)
print(alpha_df.columns)
print(google_df.columns)
print(yahoo_df.columns)

'''
fundus_df_selected = fundus_df[["", ]]
reddit_df_selected = reddit_df[["", ]]
newsapi_df_selected = newsapi_df[["", ]]
stock_df_selected = stock_df[["", ]]
'''
