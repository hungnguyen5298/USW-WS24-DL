import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import pytz

API_KEY = '923a0fe1e40f46d39ee8c9b9bc37fe8e'
BASE_URL = 'https://newsapi.org/v2/everything?'

today = datetime.today()
one_month_ago = today - timedelta(days=30)

from_date = one_month_ago.strftime('%Y-%m-%dT%H:%M:%S')
to_date = today.strftime('%Y-%m-%dT%H:%M:%S')

keyword = 'apple OR iPhone OR iPad OR MacBook OR AAPL'
parameters = {
    'q': keyword,
    'from': from_date,
    'to': to_date,
    'language': 'de',
    'pageSize': 100,
    'apiKey': API_KEY
}

response = requests.get(BASE_URL, params=parameters)
data = response.json()

if data['status'] != 'ok':
    raise Exception(f"API error: {data.get('message', 'Unknown error')}")

articles = []
berlin_tz = pytz.timezone('Europe/Berlin')

for article in data['articles']:
    if 'publishedAt' in article:
        published_utc = datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=pytz.utc)
        published_berlin = published_utc.astimezone(berlin_tz)
        formatted_date = published_berlin.strftime('%Y-%m-%d %H:%M:%S')

        articles.append({
            "Title": article.get("title"),
            "Text": article.get("description"),
            "Publisher": article['source'].get("name"),
            "URL": article.get("url"),
            "Date": formatted_date
        })

if articles:
    df = pd.DataFrame(articles)
    df = df.drop_duplicates(subset=["Title"])
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values(by='Date', ascending=True)
    df = df.reset_index(drop=True)

    os.makedirs("../project_raw_data", exist_ok=True)
    file_path = os.path.join("../project_raw_data", "filtered_full_news_newsapi_org.csv")
    df.to_csv(file_path, index=True)

else:
    print("No articles found.")
