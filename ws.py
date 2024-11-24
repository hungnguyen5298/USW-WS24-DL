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
        headlines = soup.find("h1", attrs={"class": "news-headline-title"})

        #Extrahieren
        news_list = []
        for headline in headlines:
            title = headline.get_text()
            link = headline.find('a')['href'] if headline.find('a') else None
            full_link = "https://finance.yahoo.com{link}" if link else "N/A"
            news_list.append({"title": title, "full_link": full_link})

        return news_list

    except requests.exceptions.RequestException as e:
        print(f"Fehler bei der Anfrage: {e}")
        return []

#News scrapen und ausgeben:
news = get_news(url)
for idx, article in enumerate(news, start=1):
    print(f"{idx}. {article['title']} - {article['full_link']}")

