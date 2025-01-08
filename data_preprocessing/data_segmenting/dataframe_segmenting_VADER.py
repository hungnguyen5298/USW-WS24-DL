import pandas as pd

# Beispielhafte Daten laden (ersetze dies mit deinen echten Daten)
stock_data = pd.read_csv('../../project_raw_data/stock_data_apple_full_5min.csv')
news_data = pd.read_csv('../VADER_text_pp_sentiment/news_sentiment_VADER.csv')

# Konvertiere Zeitstempel in datetime
stock_data['Datetime'] = pd.to_datetime(stock_data['Datetime'])
news_data['PublishedAt'] = pd.to_datetime(news_data['PublishedAt'])

# Timestamp shifting (jede 5 Minuten von 9:25 bis 15:55)
timestamp_shifted = (stock_data['Datetime'] - pd.Timedelta(minutes=5)).unique()
timestamp_shifted = sorted(timestamp_shifted)  # Sortiere die Timestamps zur Sicherheit

# Funktion zur Zuordnung mit neuem Limit (+4 Minuten 59 Sekunden)
def assign_shifted_timestamp(published_at, timestamp_shifted):
    for i, ts in enumerate(timestamp_shifted):
        # Überprüfe, ob PublishedAt in das Intervall passt
        if published_at < ts + pd.Timedelta(minutes=4, seconds=59):
            return ts
        # Falls nicht, nimm den nächsten Zeitstempel
        if i < len(timestamp_shifted) - 1 and published_at < timestamp_shifted[i + 1]:
            return timestamp_shifted[i + 1]
    # Falls PublishedAt nach dem letzten Zeitstempel liegt
    return None

# Wende die Funktion an
news_data['timestamp_shifted'] = news_data['PublishedAt'].apply(
    lambda x: assign_shifted_timestamp(x, timestamp_shifted)
)

# Entferne die 'PublishedAt' Spalte
news_data = news_data.drop(columns=['PublishedAt'])

# Entferne Einträge ohne gültige Zuordnung (optional)
news_data = news_data.dropna(subset=['timestamp_shifted'])

# Gruppiere nach 'timestamp_shifted' und berechne den Durchschnitt für jede Gruppe
grouped_data = news_data.groupby('timestamp_shifted').agg(
    {
        'VADER_Positive': 'mean',
        'VADER_Neutral': 'mean',
        'VADER_Negative': 'mean'
    }
)

# Füge den 'timestamp_shifted' als Spalte wieder hinzu (reset_index() macht dies)
grouped_data = grouped_data.reset_index()

# Erstelle ein vollständiges Zeitstempel-Spektrum mit allen Werten aus `timestamp_shifted`
full_timestamp_range = pd.to_datetime(timestamp_shifted)

# Reindexiere das DataFrame mit den vollständigen Zeitstempeln und setze fehlende Werte auf 0
grouped_data = grouped_data.set_index('timestamp_shifted').reindex(full_timestamp_range, fill_value=0).reset_index()

# Ändere den Namen der Index-Spalte zurück zu 'timestamp_shifted'
grouped_data = grouped_data.rename(columns={'index': 'timestamp_shifted'})

grouped_data.to_csv("agg_sentiment_VADER.csv", index=False)

print("Aggregation abgeschlossen. Die Ergebnisse wurden in 'agg_sentiment_VADER.csv' gespeichert.")