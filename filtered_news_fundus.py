from fundus import PublisherCollection, Crawler, Requires
from fundus.scraping.filter import inverse, regex_filter, lor, land
import pandas as pd
from typing import Dict, Any

crawler = Crawler(PublisherCollection.us)

filter1 = regex_filter("advertisement|podcast")
filter2 = inverse(land(regex_filter("apple")))

def date_filter(extracted: Dict[str, Any]) -> bool:
    end_date = datetime.date.today() - datetime.timedelta(weeks=1)
    start_date = end_date - datetime.timedelta(weeks=1)
    if publishing_date := extracted.get("publishing_date"):
        return not (start_date <= publishing_date.date() <= end_date)
    return True


def body_filter(extracted: Dict[str, Any]) -> bool:
    if body := extracted.get("body"):
        for word in ["apple", "iPhone", "iPad", "Macbook"]:
            if word in str(body).casefold():
                return False
    return True

articles_data = []

for article in crawler.crawl(max_articles=100, url_filter=lor(filter1, filter2)):
    articles_data.append({
        "Title": article.title,
        "Text": article.plaintext,
        "Publisher": article.publisher,
        "URL": article.html.requested_url,
        "Date": article.publishing_date,
    })

df = pd.DataFrame(articles_data)

df.to_csv('filtered_news_fundus.csv', index=True)