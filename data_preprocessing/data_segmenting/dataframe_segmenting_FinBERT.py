import pandas as pd

# Daten laden
stock_data = pd.read_csv('../../project_raw_data/stock_data_apple_full_5min.csv')
news_data = pd.read_csv('../FinBERT_text_pp_sentiment/news_sentiment_FinBERT.csv')

# Sicherstellen, dass die Zeitspalten in beiden DataFrames als Datetime-Objekte vorliegen
stock_data['Datetime'] = pd.to_datetime(stock_data['Datetime'])
news_data['PublishedAt'] = pd.to_datetime(news_data['PublishedAt'])

# DataFrame mit den Zeitstempeln aus stock_data
segments_df = pd.DataFrame({"Segment": stock_data['Datetime'].sort_values().unique()})

# Funktion zur Zuordnung der Timestamps
def assign_segment(published_at):
    # Findet den ersten Timestamp in 'stock_data', der größer oder gleich dem PublishedAt ist
    for ts in segments_df['Segment']:
        if published_at <= ts:
            return ts
    # Fallback: Wenn kein Segment gefunden wurde, None zurückgeben (nicht erwartet)
    return None

# Segment-Zuordnung zu jeder Nachricht in news_sentiment_VADER
news_data['Segment'] = news_data['PublishedAt'].apply(assign_segment)

# Gruppieren der Nachrichten basierend auf den Segmenten und Aggregieren der FinBERT-Werte
grouped_sentiments = (
    news_data.groupby("Segment")
    .agg({
        "Positive_Prob": "mean",
        "Neutral_Prob": "mean",
        "Negative_Prob": "mean"
    })
    .reset_index()
)

# Fehlende Segmente hinzufügen und mit 0 auffüllen
result = pd.merge(segments_df, grouped_sentiments, on="Segment", how="left").fillna(0)

# Ergebnis speichern
result.to_csv("agg_sentiment_FinBERT.csv", index=False)

print("Aggregation abgeschlossen. Die Ergebnisse wurden in 'agg_sentiment_FinBERT.csv' gespeichert.")