import os
import yfinance as yf
import pandas as pd

# Ticker
ticker = "AAPL"
apple_stock = yf.Ticker(ticker)

# Historische Kursdaten mit Stundenintervall für 1 Monat
historical_data = apple_stock.history(period="1mo", interval="1h")
historical_data.reset_index(inplace=True)

# Konvertiere den 'Datetime' Index in das gewünschte Format: %Y-%m-%d %H:%M:%S
historical_data['Datetime'] = historical_data['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Überprüfen, ob die CSV-Datei bereits existiert
csv_file = "../project_raw_data/stock_data_apple.csv"  # Der Pfad zur CSV-Datei

if os.path.exists(csv_file):
    # Wenn die Datei existiert, lade sie und hänge die neuen Daten an
    existing_df = pd.read_csv(csv_file)
    combined_df = pd.concat([existing_df, historical_data], ignore_index=True)
else:
    # Wenn die Datei nicht existiert, verwende nur die neuen Daten
    combined_df = historical_data

# Duplikate entfernen, falls gewünscht
combined_df = combined_df.drop_duplicates()

# Überprüfen, ob der Ordner existiert, und erstellen, falls nicht
os.makedirs(os.path.dirname(csv_file), exist_ok=True)

# In die CSV-Datei speichern
combined_df.to_csv(csv_file, index=False, encoding="utf-8")

print(f"Die Daten wurden erfolgreich in {csv_file} gespeichert.")
