from eventregistry import EventRegistry, QueryArticlesIter
import pandas as pd
from datetime import datetime
import pytz
import os

# Initialize EventRegistry with API key
er = EventRegistry(apiKey='008ffa8f-3bc2-4fbf-8aad-1d98dcc66279')

# Define the query
query = {
    "$query": {
        "$and": [
            {
                "$or": [
                    {"keyword": "Apple", "keywordLoc": "body"},
                    {"keyword": "AAPL", "keywordLoc": "body"},
                    {"keyword": "IOS", "keywordLoc": "body"},
                    {"keyword": "iPhone", "keywordLoc": "body"},
                    {"keyword": "iPad", "keywordLoc": "body"},
                    {"keyword": "MacBook", "keywordLoc": "body"}
                ]
            },
            {
                "$or": [
                    {"lang": "eng"},
                    {"lang": "deu"}
                ]
            }
        ]
    },
    "$filter": {"forceMaxDataTimeWindow": "31"}
}

# Execute query with timeout
q = QueryArticlesIter.initWithComplexQuery(query)

# Collect articles
articles = []
berlin_tz = pytz.timezone('Europe/Berlin')  # Berlin timezone

for article in q.execQuery(er, maxItems=2000):
    # Parse 'Published on' to datetime
    if 'dateTime' in article:
        # Read time from article['dateTime']
        utc_time = datetime.strptime(article['dateTime'], "%Y-%m-%dT%H:%M:%SZ")

        # Convert to Berlin time zone
        berlin_time = utc_time.replace(tzinfo=pytz.utc).astimezone(pytz.timezone("Europe/Berlin"))

        # Format time as "YYYY-MM-DD HH:MM:SS"
        berlin_time_str = berlin_time.strftime("%Y-%m-%d %H:%M:%S")
    else:
        berlin_time_str = None

    articles.append({
        "Title": article.get("title"),
        "Text": article.get("body"),
        "Publisher": article.get("source", {}).get("title"),
        "URL": article.get("url"),
        "Date": berlin_time_str
    })

# Convert to DataFrame
df = pd.DataFrame(articles)

# Remove duplicate articles based on Title across all sources
df = df.drop_duplicates(subset=["Title"])

# Convert 'Date' to datetime for sorting
df['Published on'] = pd.to_datetime(df['Date'], errors='coerce')

# Drop rows with invalid 'Published on' dates
df = df.dropna(subset=['Date'])

# Sort articles by 'Published on' in descending order
df = df.sort_values(by='Date', ascending=False)

# Save to CSV
df.to_csv("filtered_full_news_newsapi_ai.csv", index=False)
