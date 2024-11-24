'''
json --> dataframe ---> drop status, totalResult, keep articles.
Articles dictionary --> keep name, url, title, content, publishedAt --> new dataframe
f0be9f6a907649978cbca1aa52923045
'''

import requests
import pandas as pd

# url = ("https://newsapi.org/v2/everything?q=Apple&from=2024-11-20&"
#        "sortBy=popularity&apiKey=923a0fe1e40f46d39ee8c9b9bc37fe8e")

import requests
import pandas as pd

# Set display options for pandas
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

# API endpoint
keywords = "stock market AND trading AND stock price"
url = (f'https://newsapi.org/v2/everything?'
       f'q={keywords}&'
       'from=2024-11-20&'
       'sortBy=popularity&'
       'apiKey=923a0fe1e40f46d39ee8c9b9bc37fe8e')

# Send GET request
response = requests.get(url)

# Parse the JSON response
data = response.json()

# Extract 'articles' field (a list of dictionaries)
articles = data.get('articles', [])

# Create a DataFrame from the 'articles'
df = pd.DataFrame(articles)

# Select and rename required fields
df = df[['source', 'url', 'title', 'content', 'publishedAt']].copy()
df['name'] = df['source'].apply(lambda x: x['name'] if x else None)  # Extract 'name' from 'source'
df = df[['name', 'url', 'title', 'content', 'publishedAt']]  # Reorder columns

# Display the DataFrame
print(df)

df.to_csv('news_scraping.csv', index=False)

