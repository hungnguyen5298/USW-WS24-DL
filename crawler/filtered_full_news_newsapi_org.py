import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import pytz

API_KEY = '923a0fe1e40f46d39ee8c9b9bc37fe8e'
BASE_URL = 'https://newsapi.org/v2/everything?'


# Zeitrahmen für die Anfrage
today = datetime.today()
one_month_ago = today - timedelta(days=30)

from_date = one_month_ago.strftime('%Y-%m-%dT%H:%M:%S')
to_date = today.strftime('%Y-%m-%dT%H:%M:%S')

# Suchparameter
keyword = 'apple OR iPhone OR iPad OR MacBook OR AAPL'
parameters = {
    'q': keyword,
    'from': from_date,
    'to': to_date,
    'language': 'de',
    'pageSize': 100,
    'apiKey': API_KEY
}

# Anfrage an die API
response = requests.get(BASE_URL, params=parameters)
data = response.json()

# Fehlerbehandlung
if data['status'] != 'ok':
    raise Exception(f"API error: {data.get('message', 'Unknown error')}")

# Artikel extrahieren und in Berliner Zeit konvertieren
articles = []
berlin_tz = pytz.timezone('Europe/Berlin')

for article in data['articles']:
    if 'publishedAt' in article:
        published_utc = datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=pytz.utc)
        published_berlin = published_utc.astimezone(berlin_tz)
        formatted_date = published_berlin.strftime('%Y-%m-%d %H:%M:%S')

        articles.append({
            "Title": article.get("title"),
            "Text": article.get("description"),
            "Publisher": article['source'].get("name"),
            "URL": article.get("url"),
            "Date": formatted_date
        })

# Artikel verarbeiten und in CSV speichern
if articles:
    new_df = pd.DataFrame(articles)

    # Datei-Pfad für CSV
    csv_file = "../project_raw_data/filtered_full_news_newsapi_org.csv"

    if os.path.exists(csv_file):
        # Existierende Datei laden und kombinieren
        existing_df = pd.read_csv(csv_file)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        # Neue Daten als Startpunkt verwenden
        combined_df = new_df

    # Duplikate entfernen basierend auf 'URL'
    combined_df = combined_df.drop_duplicates(subset=['URL'], keep='last')

    # Datumswerte bereinigen und sortieren
    combined_df['Date'] = pd.to_datetime(combined_df['Date'], errors='coerce')
    combined_df = combined_df.dropna(subset=['Date'])
    combined_df = combined_df.sort_values(by='Date', ascending=True).reset_index(drop=True)

    # Ordner erstellen, falls nicht vorhanden, und Daten speichern
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    combined_df.to_csv(csv_file, index=False, encoding="utf-8")

    print(f"Die Daten wurden erfolgreich in {csv_file} gespeichert.")
else:
    print("Keine neuen Artikel gefunden.")