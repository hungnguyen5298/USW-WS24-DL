import feedparser
import pandas as pd
from datetime import datetime
import os

# Google News RSS-Feed URL für "Apple" (Englisch + Deutsch)
rss_url_de = "https://news.google.com/rss/search?q=apple%20OR%20AAPL&hl=de&gl=DE&ceid=DE:de"
rss_url_en = "https://news.google.com/rss/search?q=apple%20OR%20AAPL&hl=en&gl=US&ceid=US:en"

# Datum von 01.12.2024 festlegen
start_date = datetime(2024, 12, 1)

# Nachrichten extrahieren
articles = []

# Deutsche Nachrichten
feed_de = feedparser.parse(rss_url_de)
for entry in feed_de.entries:
    title = entry.title
    link = entry.link
    # Zeitstempel der Nachricht umwandeln
    timestamp = datetime(*entry.published_parsed[:6])

    # Nur Nachrichten im gewünschten Zeitraum berücksichtigen
    if timestamp >= start_date:
        # Timestamp im gewünschten Format speichern
        timestamp_formatted = timestamp.strftime('%Y-%m-%d %H:%M:%S')

        articles.append({
            "Title": title,
            "Link": link,
            "Timestamp": timestamp_formatted,
            "Language": "de"  # Für Deutsch
        })

# Englische Nachrichten
feed_en = feedparser.parse(rss_url_en)
for entry in feed_en.entries:
    title = entry.title
    link = entry.link
    # Zeitstempel der Nachricht umwandeln
    timestamp = datetime(*entry.published_parsed[:6])

    # Nur Nachrichten im gewünschten Zeitraum berücksichtigen
    if timestamp >= start_date:
        # Timestamp im gewünschten Format speichern
        timestamp_formatted = timestamp.strftime('%Y-%m-%d %H:%M:%S')

        articles.append({
            "Title": title,
            "Link": link,
            "Timestamp": timestamp_formatted,
            "Language": "en"  # Für Englisch
        })

# In DataFrame konvertieren
new_df = pd.DataFrame(articles)

# Überprüfen, ob die CSV-Datei bereits existiert
csv_file = "../project_raw_data/rss_google_news.csv"  # Der Pfad zur CSV-Datei

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
