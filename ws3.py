'''
json --> dataframe ---> drop status, totalResult, keep articles.
Articles dictionary --> keep name, url, title, content, publishedAt --> new dataframe
'''
import requests
import pandas as pd

pd.set_option('display.max_colwidth', None)
# API endpoint
url = ('https://newsapi.org/v2/everything?'
       'q=stock&'
       'from=2024-11-20&'
       'sortBy=popularity&'
       'apiKey=923a0fe1e40f46d39ee8c9b9bc37fe8e')

# url = ("https://newsapi.org/v2/everything?q=Apple&from=2024-11-20&"
#        "sortBy=popularity&apiKey=923a0fe1e40f46d39ee8c9b9bc37fe8e")

# Send GET request
response = requests.get(url)

df = pd.DataFrame(response.json())
df.to_csv('reddit_python.csv', index=False)

df['content'] = df['articles'].apply(lambda x : x.get('content'))

print(df['content'][2])