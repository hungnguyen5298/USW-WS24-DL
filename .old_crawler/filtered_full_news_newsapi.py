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
                    {"lang": "deu"},
                    {"lang": "zho"}
                ]
            }
        ]
    },
    "$filter": {"forceMaxDataTimeWindow": "31"}
}

# Execute query
q = QueryArticlesIter.initWithComplexQuery(query)

# Collect articles
articles = []
berlin_tz = pytz.timezone('Europe/Berlin')  # Berlin timezone

for article in q.execQuery(er, maxItems=100):
    # Parse 'Published on' to datetime
    try:
        published_utc = datetime.strptime(article.get("date"), '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=pytz.utc)
        # Convert UTC to Berlin timezone and extract only time
        published_berlin = published_utc.astimezone(berlin_tz).strftime('%H:%M:%S')
    except Exception as e:
        published_berlin = None  # Handle cases where date format is unexpected

    articles.append({
        "Title": article.get("title"),
        "Text": article.get("body"),
        "Publisher": article.get("source", {}).get("title"),
        "URL": article.get("url"),
        "Published on": article.get("date"),
        "Timestamp (Berlin)": published_berlin
    })

# Convert to DataFrame
df = pd.DataFrame(articles)

# Remove duplicate articles based on Title across all sources
df = df.drop_duplicates(subset=["Title"])

# Convert 'Published on' to datetime for sorting
df['Published on'] = pd.to_datetime(df['Published on'], errors='coerce')

# Drop rows with invalid 'Published on' dates
df = df.dropna(subset=['Published on'])

# Sort articles by 'Published on' in descending order
df = df.sort_values(by='Published on', ascending=False)


# Save DataFrame to a CSV file
os.makedirs("../project_raw_data", exist_ok=True)
file_path = os.path.join("../project_raw_data", "filtered_news_newsapi.csv")
df.to_csv(file_path, index=False)