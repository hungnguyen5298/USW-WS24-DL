import pandas as pd

# Importieren
vader = pd.read_csv('../data_segmenting/agg_sentiment_VADER.csv')
finbert = pd.read_csv('../data_segmenting/agg_sentiment_FINBERT.csv')

# Funktiondefinieren
def data_shifting(df):
    df['timestamp_shifted'] = pd.to_datetime(df['timestamp_shifted'])
    df['Timestamp'] = df['timestamp_shifted'] + pd.Timedelta(minutes=5)
    return df

# Zeitversatz
vader_shifted = data_shifting(vader)
finbert_shifted = data_shifting(finbert)

# Korrigeren für erste Segment täglich - 9:30

# Speichern
vader_shifted.to_csv('vader_shifted.csv', index=False)
finbert_shifted.to_csv('finbert_shifted.csv', index=False)

print("Die Sentimentdaten von VADER und FINBERT werden gespeichert werden.")