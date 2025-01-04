import os
import requests
import pandas as pd
from datetime import datetime
import pytz

url = "https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=AAPL&time_from=20241201T0000&sort=LATEST&apikey=DL068ANK2LQ03OE3"
r = requests.get(url)
data = r.json()

# JSON normalisieren
if 'feed' in data:
    new_df = pd.json_normalize(data['feed'])
else:
    new_df = pd.json_normalize(data)

# Funktion zur Konvertierung von UTC-Zeit in Berliner Zeit
def convert_to_berlin_time(utc_time_str):
    utc_time = datetime.strptime(utc_time_str, "%Y%m%dT%H%M%S")
    utc_zone = pytz.utc
    berlin_zone = pytz.timezone("Europe/Berlin")
    utc_time = utc_zone.localize(utc_time)
    berlin_time = utc_time.astimezone(berlin_zone)
    return berlin_time.strftime("%Y-%m-%d %H:%M:%S")

# Anwenden der Funktion auf die Spalte 'time_published'
if 'time_published' in new_df.columns:
    new_df['time_published'] = new_df['time_published'].apply(convert_to_berlin_time)

# Überprüfen, ob die CSV-Datei bereits existiert
csv_file = "../project_raw_data/filtered_news_alphavantage.csv"

if os.path.exists(csv_file):
    # Existierende Datei laden und kombinieren
    existing_df = pd.read_csv(csv_file)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
else:
    # Neue Daten als Startpunkt verwenden
    combined_df = new_df

# Duplikate entfernen (falls gewünscht), basierend auf der Spalte 'url' (oder einer anderen eindeutigen Spalte)
if 'url' in combined_df.columns:
    combined_df = combined_df.drop_duplicates(subset=['url'], keep='last')

# Ordner erstellen, falls nicht vorhanden
os.makedirs(os.path.dirname(csv_file), exist_ok=True)

# Kombinierte Daten in die CSV-Datei speichern
combined_df.to_csv(csv_file, index=False, encoding="utf-8")

print(f"Die Daten wurden erfolgreich in {csv_file} gespeichert.")