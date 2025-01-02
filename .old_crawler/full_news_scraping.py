from eventregistry import QueryArticlesIter, EventRegistry
import pandas as pd

# Initialize EventRegistry with API key
er = EventRegistry(apiKey='008ffa8f-3bc2-4fbf-8aad-1d98dcc66279')

# Define the query
query = {
  "$query": {
    "$and": [
      {
        "keyword": "stock price",
        "keywordLoc": "body"
      },
      {
        "keyword": "trading",
        "keywordLoc": "body"
      },
      {
        "conceptUri": "http://en.wikipedia.org/wiki/Stock_market"
      },
      {
        "$or": [
          {
            "locationUri": "http://en.wikipedia.org/wiki/United_States"
          },
          {
            "locationUri": "http://en.wikipedia.org/wiki/Germany"
          }
        ]
      }
    ]
  },
  "$filter": {
    "forceMaxDataTimeWindow": "31"
  }
}

# Execute query
q = QueryArticlesIter.initWithComplexQuery(query)

# Collect articles
articles = []
for article in q.execQuery(er, maxItems=100):
    articles.append({
        "Title": article.get("title"),
        "Source": article.get("source", {}).get("title"),
        "PublishedAt": article.get("date"),
        "URL": article.get("url"),
        "Content": article.get("body")
    })

# Convert to DataFrame
df = pd.DataFrame(articles)

# Display the DataFrame (optional)
print(df)

# Save to a CSV file
df.to_csv("stock_articles.csv", index=False)
