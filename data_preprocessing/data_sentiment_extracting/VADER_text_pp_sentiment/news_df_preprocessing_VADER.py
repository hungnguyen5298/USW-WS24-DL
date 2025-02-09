import pandas as pd
from bs4 import BeautifulSoup
import nltk
import re
import emoji
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

# Stelle sicher, dass die Stopwörter verfügbar sind
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# Data importieren
news_df = pd.read_csv('../../data_merging/news_df.csv')

# Filtern nach Periode von 01-12.2024 - 03.01.2025
news_df['PublishedAt'] = pd.to_datetime(news_df['PublishedAt'])
start_date = pd.to_datetime('2024-12-04')
end_date = pd.to_datetime('2025-01-07')
news_date_selected = news_df[(news_df['PublishedAt'] >= start_date) & (news_df['PublishedAt'] <= end_date)]

# Text preprocessing
news_input = news_date_selected.copy()
# Text preprocessing
def normalize_text(text, lang='en'):
    # 1. HTML-Tags entfernen
    text = BeautifulSoup(text, "html.parser").get_text()

    # 2. Emoji entfernen
    text = emoji.replace_emoji(text, replace='')  # Ersetzt Emojis durch einen leeren String

    # 3. Kleinbuchstaben umwandeln
    text = text.lower()

    # 4. Sonderzeichen entfernen
    text = re.sub(r'[^a-zA-ZäöüÄÖÜß0-9\s$€]', '', text)

    # 5. Tokenisierung
    tokens = word_tokenize(text)

    # 6. Stoppwörter entfernen (für Englisch und Deutsch)
    if lang == 'en':
        stop_words = set(stopwords.words('english'))
    elif lang == 'de':
        stop_words = set(stopwords.words('german'))
    else:
        stop_words = set(stopwords.words('english')).union(set(stopwords.words('german')))

    filtered_tokens = [word for word in tokens if word not in stop_words]

    # 7. Lemmatisierung (für Englisch mit WordNetLemmatizer, für Deutsch mit NLTK)
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    # Rückgabe der Liste von tokenisierten Wörtern
    return lemmatized_tokens

# Fehlende Werte in 'Title' und 'Content' durch leere Strings ersetzen
news_input['Title'] = news_input['Title'].fillna("")
news_input['Content'] = news_input['Content'].fillna("")

# Normalisierung der 'Title' und 'Content' Spalten anwenden
news_input['Title'] = news_input['Title'].apply(lambda x: normalize_text(x, lang='en'))
news_input['Content'] = news_input['Content'].apply(lambda x: normalize_text(x, lang='en'))

# Neuer DataFrame mit den normalisierten Spalten 'Title', 'Content' und 'PublishedAt'
text_preprocessed = news_input[['Title', 'Content', 'PublishedAt']]

# Speichern des neuen DataFrames
text_preprocessed.to_csv('text_preprocessed_VADER.csv', index=False, header=True)

# Sentiment extrahieren
sia = SentimentIntensityAnalyzer()

# Funktion, um Sentiment-Scores zu berechnen
def calculate_sentiment(text):
    sentiment_scores = sia.polarity_scores(text)
    return sentiment_scores['pos'], sentiment_scores['neu'], sentiment_scores['neg']

# Berechnung der Sentiments für die Kombination von 'Title' und 'Content'
def compute_sentiment_for_news(row):
    # Kombiniere Titel und Inhalt zu einem Textstring
    combined_text = ' '.join(row['Title']) + ' ' + ' '.join(row['Content'])

    # Berechne die Sentiment-Werte für den kombinierten Text
    pos, neu, neg = calculate_sentiment(combined_text)

    return pd.Series([pos, neu, neg])


# Berechne die Sentiment Scores für alle Nachrichten
news_input[['VADER_Positive', 'VADER_Neutral', 'VADER_Negative']] = news_input.apply(compute_sentiment_for_news, axis=1)

# Speichern der DataFrame mit Sentiment-Werten und 'PublishedAt' Timestamp
news_sentiment = news_input[['PublishedAt', 'VADER_Positive', 'VADER_Neutral', 'VADER_Negative']]

# Optional: Speichern des neuen DataFrames mit Sentiment-Werten in eine CSV-Datei
news_sentiment.to_csv('news_sentiment_VADER.csv', index=False, header=True)