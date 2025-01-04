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

# Fundus-Daten
fundus_selected = fundus_df[['Title', 'Text', 'Date']].rename(columns={'Text': 'Content', 'Date': 'PublishedAt'})

# Reddit-Daten (Content und Comments kombinieren)
reddit_selected = reddit_df[['Title', 'Content', 'Comments', 'Published_at']].assign(
    Content=reddit_df['Content'] + " " + reddit_df['Comments']
)[['Title', 'Content', 'Published_at']].rename(columns={'Published_at': 'PublishedAt'})

# NewsAPI-Daten
newsapi_selected = newsapi_df[['Title', 'Text', 'Date']].rename(columns={'Text': 'Content', 'Date': 'PublishedAt'})

# Alpha-Daten
alpha_selected = alpha_df[['title', 'summary', 'time_published']].rename(columns={
    'title': 'Title', 'summary': 'Content', 'time_published': 'PublishedAt'
})

# Google-Daten (Leerer Content)
google_selected = google_df[['Title', 'Timestamp']].assign(Content=" ").rename(columns={'Timestamp': 'PublishedAt'})

# Yahoo-Daten
yahoo_selected = yahoo_df[['Title', 'Summary', 'Published_at']].rename(columns={
    'Summary': 'Content', 'Published_at': 'PublishedAt'
})

# DataFrames kombinieren
news_df = pd.concat([
    fundus_selected,
    reddit_selected,
    newsapi_selected,
    alpha_selected,
    google_selected,
    yahoo_selected
], ignore_index=True)

news_df.to_csv('news_df.csv', index=False, header=True)

print(f"Die Daten wurden erfolgreich gespeichert.")