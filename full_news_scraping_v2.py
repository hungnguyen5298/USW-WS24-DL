from fundus import PublisherCollection, Crawler
import pandas as pd
#CNBC, BoersenZeitung
# initialize the crawler for The New Yorker
crawler = Crawler(PublisherCollection.us.CNBC)

articles_data = []

# Crawl 10 articles
for article in crawler.crawl(max_articles=10):
    # Extrahiere relevante Attribute
    articles_data.append({
        "Title": article.title,
        "Text": article.plaintext,
        "Date": article.publishing_date,
    })

# Konvertiere die Artikel-Daten in einen DataFrame
df = pd.DataFrame(articles_data)

# Speichere die Daten als CSV
df.to_csv('test.csv', index=False)

print("Artikel wurden erfolgreich gespeichert!")