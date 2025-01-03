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
    df = pd.json_normalize(data['feed'])
else:
    df = pd.json_normalize(data)

# Funktion zur Konvertierung von UTC-Zeit in Berliner Zeit
def convert_to_berlin_time(utc_time_str):
    utc_time = datetime.strptime(utc_time_str, "%Y%m%dT%H%M%S")
    utc_zone = pytz.utc
    berlin_zone = pytz.timezone("Europe/Berlin")
    utc_time = utc_zone.localize(utc_time)
    berlin_time = utc_time.astimezone(berlin_zone)
    return berlin_time.strftime("%Y-%m-%d %H:%M:%S")

# Anwenden der Funktion auf die Spalte 'time_published'
if 'time_published' in df.columns:
    df['time_published'] = df['time_published'].apply(convert_to_berlin_time)

# In CSV speichern
os.makedirs("../project_raw_data", exist_ok=True)
file_path = os.path.join("../project_raw_data", "filtered_news_alphavantage.csv")
df.to_csv(file_path, index=True)