import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Versuche, die CSV-Datei zu laden
try:
    news_df = pd.read_csv('news_df.csv')
except FileNotFoundError:
    print("Die Datei 'news_df.csv' wurde nicht gefunden.")
    exit()

# Filtern nach Periode von 01-12.2024 - 03.01.2025
news_df['PublishedAt'] = pd.to_datetime(news_df['PublishedAt'])
start_date = pd.to_datetime('2024-12-01')
end_date = pd.to_datetime('2025-01-04')
news_date_selected = news_df[(news_df['PublishedAt'] >= start_date) & (news_df['PublishedAt'] <= end_date)]

# Text preprocessing
news_input = news_date_selected.copy()

# Ersetzen von NaN-Werten in 'Title' und 'Content' durch leere Strings
news_input['Title'] = news_input['Title'].fillna("")
news_input['Content'] = news_input['Content'].fillna("")

# Kombiniere Titel und Inhalt
def combine_title_and_content(row):
    title = str(row.get('Title', ''))  # Sicherstellen, dass der Wert als String behandelt wird
    content = str(row.get('Content', ''))  # Sicherstellen, dass der Wert als String behandelt wird
    return title + " " + content

# Anwenden der Kombination auf alle Zeilen
news_input['combined_text'] = news_input.apply(combine_title_and_content, axis=1)

# Pfad zum Modell und Tokenizer
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)

# Falls GPU vorhanden ist, auf die GPU verschieben
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Funktion zur Berechnung des Sentiments mit FinBERT
def calculate_sentiment_with_finbert(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Softmax auf die Ergebnisse anwenden, um Wahrscheinlichkeiten zu erhalten
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Extrahiere die Wahrscheinlichkeit für jede Klasse (negative, neutral, positive)
    sentiment_probabilities = probs.squeeze().cpu().numpy()

    return sentiment_probabilities

# Berechne Sentiment-Wahrscheinlichkeiten für jede Nachricht
news_input[['Negative_Prob', 'Neutral_Prob', 'Positive_Prob']] = news_input['combined_text'].apply(
    lambda x: pd.Series(calculate_sentiment_with_finbert(x))
)

# Optional: Speichern der Ergebnisse
news_input[['PublishedAt', 'Negative_Prob', 'Neutral_Prob', 'Positive_Prob']].to_csv(
    'news_sentiment_FinBERT.csv', index=False
)
