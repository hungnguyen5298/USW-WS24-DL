import pandas as pd
from tensorflow.python.ops.state_ops import assign

# Laden des Datensatzes
data = pd.read_csv('../VADER_text_pp_sentiment/news_sentiment_VADER.csv')
data['PublishedAt'] = pd.to_datetime(data['PublishedAt'])

# Definiere den Zeitraum für die Analyse
start_date = pd.to_datetime('2024-12-03')
end_date = pd.to_datetime('2025-01-03')
news_date_selected = data[(data['PublishedAt'] >= start_date) & (data['PublishedAt'] <= end_date)].copy()
import pandas as pd

# Liste für die Zeitstempel erstellen
time_range = []

# Für jedes Datum im Zeitraum die stündlichen Zeitstempel von 09:30 bis 16:00 generieren
current_date = start_date
while current_date <= end_date:
    # Überprüfen, ob der aktuelle Tag ein Werktag ist (Montag bis Freitag)
    if current_date.weekday() < 5:  # Montag bis Freitag (0-4)
        day_start = pd.Timestamp(current_date.strftime('%Y-%m-%d') + ' 09:30:00')
        day_end = pd.Timestamp(current_date.strftime('%Y-%m-%d') + ' 16:00:00')

        # Zeitstempel von 09:30 bis 16:00 stündlich erzeugen
        daily_range = pd.date_range(start=day_start, end=day_end, freq='h')

        # Die täglichen Zeitstempel zur Liste hinzufügen
        time_range.extend(daily_range)

    # Nächster Tag
    current_date += pd.Timedelta(days=1)

# Erstelle eine leere Liste für die Gruppen
groups = []

# Durchlaufe jedes PublishedAt im DataFrame und ordne es dem passenden Zeitbereich zu
for idx, row in news_date_selected.iterrows():
    published_at = row['PublishedAt']
    assigned = False  # Flag, um zu überprüfen, ob ein passender Zeitraum gefunden wurde

    # Suche den passenden Zeitbereich in time_range
    for i in range(len(time_range) - 1):  # Vergleiche mit den benachbarten Zeitstempeln
        if time_range[i] <= published_at < time_range[i + 1]:
            groups.append(time_range[
                              i + 1])  # Wenn es zwischen time_range[i] und time_range[i + 1] liegt, dann ordne den oberen Grenzwert zu
            assigned = True
            break  # Wenn der passende Bereich gefunden wurde, beende die Schleife

    if not assigned:  # Wenn kein passender Zeitraum gefunden wurde, füge None oder NaN hinzu
        groups.append(None)  # Hier kannst du auch `pd.NaT` oder einen anderen Wert wie `NaN` verwenden

# Überprüfen, ob die Länge von 'groups' gleich der Länge von 'news_date_selected' ist
if len(groups) == len(news_date_selected):
    # Füge die Zeitstempel als neue Spalte in den DataFrame ein
    news_date_selected['TimeGroup'] = groups
    # Anzeige der gruppierten Daten
    print(news_date_selected)
else:
    print(
        f"Fehler: Die Anzahl der Zeitgruppen ({len(groups)}) stimmt nicht mit der Anzahl der Datensätze ({len(news_date_selected)}) überein.")

# Gruppieren nach TimeGroup und Berechnung des Mittelwerts für die Sentiment-Spalten
result = news_date_selected.groupby('TimeGroup').agg({
    'VADER_Negative': 'mean',
    'VADER_Neutral': 'mean',
    'VADER_Positive': 'mean'
}).reset_index()

# Umbenennen der Spalten für bessere Lesbarkeit
result.columns = ['TimeGroup', 'mean_negative', 'mean_neutral', 'mean_positive']

result.to_csv('news_sentiment_VADER_grouped.csv', index=False)