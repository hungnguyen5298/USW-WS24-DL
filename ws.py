import requests
from bs4 import BeautifulSoup

#1.Test mit URL
url = "https://finance.yahoo.com/news/junk-bonds-risk-emerging-markets-133000454.html"

def get_news(url):
    try:
        #HTTP-Anfrage schicken
        response = requests.get(url)
        response.raise_for_status()

        #Inhalt parsen
        soup = BeautifulSoup(response.text, "lxml")

        #Inhalt-Container finden
        headlines = soup.find("h3", attrs={"class": "news-headline-title"})

        #Extrahieren
        news_list = []
        for headline in headlines:
            title = headline.get_text()
            link = headline.find()
            //TODO
