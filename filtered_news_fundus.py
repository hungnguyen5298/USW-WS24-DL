import os
from fundus import PublisherCollection, Crawler, Requires
from fundus.scraping.filter import inverse, regex_filter, lor, land
import pandas as pd
from typing import Dict, Any
from datetime import datetime, timedelta

crawler = Crawler(PublisherCollection.us, PublisherCollection.uk, PublisherCollection.de, PublisherCollection.cn)

filter1 = regex_filter("advertisement|podcast")
filter2 = inverse(land(regex_filter("apple")))

'''
def date_filter(extracted: Dict[str, Any]) -> bool:
    end_date = datetime.date.today() - datetime.timedelta(weeks=1)
    start_date = end_date - datetime.timedelta(weeks=1)
    if publishing_date := extracted.get("publishing_date"):
        return not (start_date <= publishing_date.date() <= end_date)
    return True
'''

def date_filter(extracted: Dict[str, Any]) -> bool:
    # Define the start and end dates for the filter
    today = datetime.today()
    one_month_ago = today - timedelta(days=30)

    # Check the publishing date
    if publishing_date := extracted.get("publishing_date"):
        return one_month_ago <= publishing_date <= today

    # Exclude articles with no publishing date
    return False

def body_filter(extracted: Dict[str, Any]) -> bool:
    if body := extracted.get("body"):
        for word in ["apple", "iPhone", "iPad", "Macbook", "AAPL"]:
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
        "Date": article.publishing_date.strftime('%Y-%m-%d %H:%M:%S'),
    })

df = pd.DataFrame(articles_data)

os.makedirs("project_raw_data", exist_ok=True)
file_path = os.path.join("project_raw_data", "filtered_news_fundus.csv")
df.to_csv(file_path, index=True)