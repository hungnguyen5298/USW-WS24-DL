import pandas as pd

# Laden des Datensatzes
data = pd.read_csv('news_df.csv')
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


