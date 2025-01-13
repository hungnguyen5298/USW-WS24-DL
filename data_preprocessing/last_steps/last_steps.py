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
split_index_stock_train = int(len(stock) * 0.7)
split_index_stock_val = split_index_stock_train + int(len(stock) * 0.2)
stock_X = stock.drop(columns=['Close'])
stock_y = stock['Close']
stock_X_train, stock_X_val, stock_X_test = stock_X[:split_index_stock_train], stock_X[split_index_stock_train:split_index_stock_val], stock_X[split_index_stock_val:]
stock_y_train, stock_y_val, stock_y_test = stock_y[:split_index_stock_train], stock_y[split_index_stock_train:split_index_stock_val], stock_y[split_index_stock_val:]

split_index_vader_train = int(len(vader) * 0.7)
split_index_vader_val = split_index_vader_train + int(len(vader) * 0.2)
vader_X = vader.drop(columns=['Close'])
vader_y = vader['Close']
vader_X_train, vader_X_val, vader_X_test = vader_X[:split_index_vader_train], vader_X[split_index_vader_train:split_index_vader_val], vader_X[split_index_vader_val:]
vader_y_train, vader_y_val, vader_y_test = vader_y[:split_index_vader_train], vader_y[split_index_vader_train:split_index_vader_val], vader_y[split_index_vader_val:]

split_index_finbert_train = int(len(finbert) * 0.7)
split_index_finbert_val = split_index_finbert_train + int(len(finbert) * 0.2)
finbert_X = finbert.drop(columns=['Close'])
finbert_y = finbert['Close']
finbert_X_train, finbert_X_val, finbert_X_test = finbert_X[:split_index_finbert_train], finbert_X[split_index_finbert_train:split_index_finbert_val], finbert_X[split_index_finbert_val:]
finbert_y_train, finbert_y_val, finbert_y_test = finbert_y[:split_index_finbert_train], finbert_y[split_index_finbert_train:split_index_finbert_val], finbert_y[split_index_finbert_val:]

# MinMax-Skalierung nur auf numerische Features

# Stock-Daten skalieren
scaler_X_stock_train = MinMaxScaler(feature_range=(0, 1))
scaler_X_stock_val = MinMaxScaler(feature_range=(0, 1))
scaler_X_stock_test = MinMaxScaler(feature_range=(0, 1))

numerical_columns_stock = stock_X_train.select_dtypes(include=['float64', 'int64']).columns
stock_X_train[numerical_columns_stock] = scaler_X_stock_train.fit_transform(stock_X_train[numerical_columns_stock])
stock_X_val[numerical_columns_stock] = scaler_X_stock_val.fit_transform(stock_X_val[numerical_columns_stock])
stock_X_test[numerical_columns_stock] = scaler_X_stock_test.fit_transform(stock_X_test[numerical_columns_stock])

scaler_y_stock_train = MinMaxScaler(feature_range=(0, 1))
scaler_y_stock_val = MinMaxScaler(feature_range=(0, 1))
scaler_y_stock_test = MinMaxScaler(feature_range=(0, 1))

stock_y_train = scaler_y_stock_train.fit_transform(stock_y_train.values.reshape(-1, 1))
stock_y_val = scaler_y_stock_val.fit_transform(stock_y_val.values.reshape(-1, 1))
stock_y_test = scaler_y_stock_test.fit_transform(stock_y_test.values.reshape(-1, 1))

# Vader-Daten skalieren
scaler_X_vader_train = MinMaxScaler(feature_range=(0, 1))
scaler_X_vader_val = MinMaxScaler(feature_range=(0, 1))
scaler_X_vader_test = MinMaxScaler(feature_range=(0, 1))

numerical_columns_vader = vader_X_train.select_dtypes(include=['float64', 'int64']).columns
vader_X_train[numerical_columns_vader] = scaler_X_vader_train.fit_transform(vader_X_train[numerical_columns_vader])
vader_X_val[numerical_columns_vader] = scaler_X_vader_val.fit_transform(vader_X_val[numerical_columns_vader])
vader_X_test[numerical_columns_vader] = scaler_X_vader_test.fit_transform(vader_X_test[numerical_columns_vader])

scaler_y_vader_train = MinMaxScaler(feature_range=(0, 1))
scaler_y_vader_val = MinMaxScaler(feature_range=(0, 1))
scaler_y_vader_test = MinMaxScaler(feature_range=(0, 1))

vader_y_train = scaler_y_vader_train.fit_transform(vader_y_train.values.reshape(-1, 1))
vader_y_val = scaler_y_vader_val.fit_transform(vader_y_val.values.reshape(-1, 1))
vader_y_test = scaler_y_vader_test.fit_transform(vader_y_test.values.reshape(-1, 1))

# FinBERT-Daten skalieren
scaler_X_finbert_train = MinMaxScaler(feature_range=(0, 1))
scaler_X_finbert_val = MinMaxScaler(feature_range=(0, 1))
scaler_X_finbert_test = MinMaxScaler(feature_range=(0, 1))

numerical_columns_finbert = finbert_X_train.select_dtypes(include=['float64', 'int64']).columns
finbert_X_train[numerical_columns_finbert] = scaler_X_finbert_train.fit_transform(finbert_X_train[numerical_columns_finbert])
finbert_X_val[numerical_columns_finbert] = scaler_X_finbert_val.fit_transform(finbert_X_val[numerical_columns_finbert])
finbert_X_test[numerical_columns_finbert] = scaler_X_finbert_test.fit_transform(finbert_X_test[numerical_columns_finbert])

scaler_y_finbert_train = MinMaxScaler(feature_range=(0, 1))
scaler_y_finbert_val = MinMaxScaler(feature_range=(0, 1))
scaler_y_finbert_test = MinMaxScaler(feature_range=(0, 1))

finbert_y_train = scaler_y_finbert_train.fit_transform(finbert_y_train.values.reshape(-1, 1))
finbert_y_val = scaler_y_finbert_val.fit_transform(finbert_y_val.values.reshape(-1, 1))
finbert_y_test = scaler_y_finbert_test.fit_transform(finbert_y_test.values.reshape(-1, 1))

# Sequenzen unterteilen
def create_sequences_numpy(X, y, window_size):
    sequences_X = []
    sequences_y = []
    for i in range(len(X) - window_size):
        # Hole eine Sequenz von Größe `window_size`
        sequences_X.append(X[i:i + window_size])
        # Zielwert ist der Wert direkt nach der Sequenz
        sequences_y.append(y[i + window_size])
    return np.array(sequences_X), np.array(sequences_y)

window_size = 3
# Sequenzen für Stock-Daten
X_train_stock, y_train_stock = create_sequences_numpy(stock_X_train, stock_y_train, window_size)
X_val_stock, y_val_stock = create_sequences_numpy(stock_X_val, stock_y_val, window_size)
X_test_stock, y_test_stock = create_sequences_numpy(stock_X_test, stock_y_test, window_size)

# Sequenzen für Vader-Daten
X_train_vader, y_train_vader = create_sequences_numpy(vader_X_train, vader_y_train, window_size)
X_val_vader, y_val_vader = create_sequences_numpy(vader_X_val, vader_y_val, window_size)
X_test_vader, y_test_vader = create_sequences_numpy(vader_X_test, vader_y_test, window_size)

# Sequenzen für FinBERT-Daten
X_train_finbert, y_train_finbert = create_sequences_numpy(finbert_X_train, finbert_y_train, window_size)
X_val_finbert, y_val_finbert = create_sequences_numpy(finbert_X_val, finbert_y_val, window_size)
X_test_finbert, y_test_finbert = create_sequences_numpy(finbert_X_test, finbert_y_test, window_size)

# Speichern der Sequenzen mit angepasstem Pfad
def save_to_npy(output_dir, X, y, file_prefix):
    np.save(os.path.join(output_dir, f'{file_prefix}_X.npy'), X)
    np.save(os.path.join(output_dir, f'{file_prefix}_y.npy'), y)
    print(f'Dateien gespeichert: {file_prefix}_X.npy, {file_prefix}_y.npy im Ordner {output_dir}')

# Stock-Daten speichern
save_to_npy('../../model/HLOCV',X_train_stock, y_train_stock, 'stock_train')
save_to_npy('../../model/HLOCV', X_val_stock, y_val_stock, 'stock_val')
save_to_npy('../../model/HLOCV',X_test_stock, y_test_stock, 'stock_test')

# Vader-Daten speichern
save_to_npy('../../model/HLOCV_VADER',X_train_vader, y_train_vader, 'vader_train')
save_to_npy('../../model/HLOCV_VADER', X_val_vader, y_val_vader, 'vader_val')
save_to_npy('../../model/HLOCV_VADER', X_test_vader, y_test_vader, 'vader_test')

# FinBERT-Daten speichern
save_to_npy('../../model/HLOCV_FinBERT', X_train_finbert, y_train_finbert, 'finbert_train')
save_to_npy('../../model/HLOCV_FinBERT', X_val_finbert, y_val_finbert, 'finbert_val')
save_to_npy('../../model/HLOCV_FinBERT', X_test_finbert, y_test_finbert, 'finbert_test')