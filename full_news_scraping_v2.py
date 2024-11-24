from fundus import PublisherCollection, Crawler
import pandas as pd

crawler = Crawler(PublisherCollection.us.CNBC)

articles_data = []

for article in crawler.crawl(max_articles=100):
    articles_data.append({
        "Title": article.title,
        "Text": article.plaintext,
        "Publisher": article.publisher,
        "URL": article.html.requested_url,
        "Date": article.publishing_date,
    })

df = pd.DataFrame(articles_data)

df.to_csv('News_CNBC.csv', index=False)

print("Artikel wurden erfolgreich gespeichert!")