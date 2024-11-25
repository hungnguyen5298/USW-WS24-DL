from eventregistry import EventRegistry, QueryArticlesIter
import pandas as pd

# Initialize EventRegistry with API key
er = EventRegistry(apiKey='008ffa8f-3bc2-4fbf-8aad-1d98dcc66279')

# Define the query
query = {
  "$query": {
    "$and": [
      {
        "$or": [
          {"keyword": "Apple",
            "keywordLoc": "body"},
          {"keyword": "AAPL",
            "keywordLoc": "body"},
          {"keyword": "IOS",
            "keywordLoc": "body"},
          {"keyword": "iPhone",
            "keywordLoc": "body"},
          {"keyword": "iPad",
            "keywordLoc": "body"},
          {"keyword": "MacBook",
            "keywordLoc": "body"}
        ]
      },
      {"locationUri": "http://en.wikipedia.org/wiki/United_States"},
      {"lang": "eng"}
    ]
  },
    "$filter": {"forceMaxDataTimeWindow": "31"}
}

# Execute query
q = QueryArticlesIter.initWithComplexQuery(query)

# Collect articles
articles = []
for article in q.execQuery(er, maxItems=100):
    articles.append({
        "Title": article.get("title"),
        "Text": article.get("body"),
        "Publisher": article.get("source", {}).get("title"),
        "URL": article.get("url"),
        "Published on": article.get("date")
    })

# Convert to DataFrame
df = pd.DataFrame(articles)

# Display the DataFrame
print(df)

# Save DataFrame to a CSV file
df.to_csv("filtered_full_news_newsapi.csv", index=False)
