import pandas as pd
import pandas_ta as ta

# Daten laden
data = pd.read_csv('../../project_raw_data/stock_data_apple_max.csv')

# Berechne die Volatilität (Standardabweichung der Schlusskurse)
data['Volatility'] = data['Close'].rolling(window=14).std()

# Berechne den RSI
data['RSI'] = ta.rsi(data['Close'], length=14)

# Berechne den OBV
data['OBV'] = ta.obv(data['Close'], data['Volume'])

# Berechne den ATR
data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)

# Berechne den SMA (Simple Moving Average)
data['SMA_14'] = data['Close'].rolling(window=14).mean()

# Berechne den EMA (Exponential Moving Average)
data['EMA_14'] = ta.ema(data['Close'], length=14)

# Berechne den Rate of Change (ROC)
data['ROC_14'] = ta.roc(data['Close'], length=14)

# Profitchange calculating and labeling
data['Change_Close'] = data['Close'].diff()

data['percent_change_close'] = data['Close'].pct_change()

data['Profit_Trend_Label'] = (data['percent_change_close'] > 0).astype(int)
# Labels erstellen: "steigt" und "sinkt"
#data['Profit_Trend_Label'] = data['percent_change_close'].apply(lambda x: 'steigt' if x > 0 else 'sinkt')

# Bereinige unnötige Spalten
data.drop(columns=['Dividends', 'Stock Splits'], inplace=True)

# Timestamp-Spalte umbenennen
data.rename(columns={'Date': 'Timestamp'}, inplace=True)

data = data[15:]

# Daten speichern
data.to_csv('stock_data_apple_indicators_longer.csv', index=False)
