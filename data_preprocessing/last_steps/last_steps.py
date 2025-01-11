import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Daten importieren
stock = pd.read_csv('../stock_data_feature_engineering/stock_data_apple_indicators.csv')
vader = pd.read_csv('../data_merging/vader_stock_joined.csv')
finbert = pd.read_csv('../data_merging/finbert_stock_joined.csv')

# Nach Datum sortieren
stock = stock.sort_values(by='Timestamp', ascending=True)
vader = vader.sort_values(by='Timestamp', ascending=True)
finbert = finbert.sort_values(by='Timestamp', ascending=True)

# Drop Timestamp
stock = stock.drop(columns=['Timestamp'])
vader = vader.drop(columns=['Timestamp'])
finbert = finbert.drop(columns=['Timestamp'])

# Feature und Target auswählen und Daten splitten
split_index_stock = int(len(stock) * 0.8)
stock_X = stock.drop(columns=['Profit_Trend_Label'])
stock_y = stock['Profit_Trend_Label']
stock_X_train, stock_X_test = stock_X[:split_index_stock], stock_X[split_index_stock:]
stock_y_train, stock_y_test = stock_y[:split_index_stock], stock_y[split_index_stock:]

split_index_vader = int(len(vader) * 0.8)
vader_X = vader.drop(columns=['Profit_Trend_Label'])
vader_y = vader['Profit_Trend_Label']
vader_X_train, vader_X_test = vader_X[:split_index_vader], vader_X[split_index_vader:]
vader_y_train, vader_y_test = vader_y[:split_index_vader], vader_y[split_index_vader:]

split_index_finbert = int(len(finbert) * 0.8)
finbert_X = finbert.drop(columns=['Profit_Trend_Label'])
finbert_y = finbert['Profit_Trend_Label']
finbert_X_train, finbert_X_test = finbert_X[:split_index_finbert], finbert_X[split_index_finbert:]
finbert_y_train, finbert_y_test = finbert_y[:split_index_finbert], finbert_y[split_index_finbert:]

# MinMax-Skalierung nur auf numerische Features
scaler = MinMaxScaler(feature_range=(0, 1))

# Stock-Daten skalieren
numerical_columns_stock = stock_X_train.select_dtypes(include=['float64', 'int64']).columns
stock_X_train[numerical_columns_stock] = scaler.fit_transform(stock_X_train[numerical_columns_stock])
stock_X_test[numerical_columns_stock] = scaler.transform(stock_X_test[numerical_columns_stock])

# Vader-Daten skalieren
numerical_columns_vader = vader_X_train.select_dtypes(include=['float64', 'int64']).columns
vader_X_train[numerical_columns_vader] = scaler.fit_transform(vader_X_train[numerical_columns_vader])
vader_X_test[numerical_columns_vader] = scaler.transform(vader_X_test[numerical_columns_vader])

# FinBERT-Daten skalieren
numerical_columns_finbert = finbert_X_train.select_dtypes(include=['float64', 'int64']).columns
finbert_X_train[numerical_columns_finbert] = scaler.fit_transform(finbert_X_train[numerical_columns_finbert])
finbert_X_test[numerical_columns_finbert] = scaler.transform(finbert_X_test[numerical_columns_finbert])

# Sequenzen erstellen
def create_sequences(X, y, window_size):
    sequences_X = []
    sequences_y = []
    for i in range(len(X) - window_size):
        # Hole eine Sequenz von Größe `window_size`
        sequences_X.append(X[i:i + window_size].values)
        # Zielwert ist der Wert direkt nach der Sequenz
        sequences_y.append(y.iloc[i + window_size])
    return np.array(sequences_X), np.array(sequences_y)

window_size = 3
# Sequenzen für Stock-Daten
X_train_stock, y_train_stock = create_sequences(stock_X_train.reset_index(drop=True), stock_y_train.reset_index(drop=True), window_size)
X_test_stock, y_test_stock = create_sequences(stock_X_test.reset_index(drop=True), stock_y_test.reset_index(drop=True), window_size)

# Sequenzen für Vader-Daten
X_train_vader, y_train_vader = create_sequences(vader_X_train.reset_index(drop=True), vader_y_train.reset_index(drop=True), window_size)
X_test_vader, y_test_vader = create_sequences(vader_X_test.reset_index(drop=True), vader_y_test.reset_index(drop=True), window_size)

# Sequenzen für FinBERT-Daten
X_train_finbert, y_train_finbert = create_sequences(finbert_X_train.reset_index(drop=True), finbert_y_train.reset_index(drop=True), window_size)
X_test_finbert, y_test_finbert = create_sequences(finbert_X_test.reset_index(drop=True), finbert_y_test.reset_index(drop=True), window_size)

# Speichern der Sequenzen mit angepasstem Pfad
def save_to_npy(output_dir, X, y, file_prefix):
    np.save(os.path.join(output_dir, f'{file_prefix}_X.npy'), X)
    np.save(os.path.join(output_dir, f'{file_prefix}_y.npy'), y)
    print(f'Dateien gespeichert: {file_prefix}_X.npy, {file_prefix}_y.npy im Ordner {output_dir}')

# Stock-Daten speichern
save_to_npy('../../model/HLOCV',X_train_stock, y_train_stock, 'stock_train')
save_to_npy('../../model/HLOCV',X_test_stock, y_test_stock, 'stock_test')

# Vader-Daten speichern
save_to_npy('../../model/HLOCV_VADER',X_train_vader, y_train_vader, 'vader_train')
save_to_npy('../../model/HLOCV_VADER', X_test_vader, y_test_vader, 'vader_test')

# FinBERT-Daten speichern
save_to_npy('../../model/HLOCV_FinBERT', X_train_finbert, y_train_finbert, 'finbert_train')
save_to_npy('../../model/HLOCV_FinBERT', X_test_finbert, y_test_finbert, 'finbert_test')