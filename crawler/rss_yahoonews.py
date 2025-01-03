import feedparser
import pandas as pd
from datetime import datetime
import pytz
import os

# Yahoo Finance RSS-Feed für Apple
rss_url = "https://finance.yahoo.com/rss/2.0/headline?s=AAPL&region=US&lang=en-US"

# Funktion zur Konvertierung von UTC-Zeit in Berliner Zeit
def convert_to_berlin_time(published_time):
    # Zeitstempel parsen
    utc_time = datetime.strptime(published_time, "%a, %d %b %Y %H:%M:%S %z")
    # In Berliner Zeit umwandeln
    berlin_zone = pytz.timezone("Europe/Berlin")
    berlin_time = utc_time.astimezone(berlin_zone)
    return berlin_time.strftime("%Y-%m-%d %H:%M:%S")

# Daten extrahieren
articles = []
feed = feedparser.parse(rss_url)

for entry in feed.entries:
    article = {
        "Title": entry.title,
        "Link": entry.link,
        "Published_at": convert_to_berlin_time(entry.published),
        "Summary": entry.summary,
    }
    articles.append(article)

# In DataFrame konvertieren
new_df = pd.DataFrame(articles)

# Überprüfen, ob die CSV-Datei bereits existiert
csv_file = "../project_raw_data/rss_yahoofinance.csv"  # Der Pfad zur CSV-Datei

if os.path.exists(csv_file):
    # Wenn die Datei existiert, lade sie und hänge die neuen Nachrichten an
    existing_df = pd.read_csv(csv_file)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
else:
    # Wenn die Datei nicht existiert, verwende nur die neuen Nachrichten
    combined_df = new_df

# Duplikate entfernen (falls gewünscht), basierend auf dem Link
combined_df = combined_df.drop_duplicates(subset=["Link"], keep="last")

# Überprüfen, ob der Ordner existiert, und erstellen, falls nicht
os.makedirs(os.path.dirname(csv_file), exist_ok=True)

# In die CSV-Datei speichern
combined_df.to_csv(csv_file, index=False, encoding="utf-8")

print(f"Die RSS-Daten wurden erfolgreich in {csv_file} gespeichert.")
