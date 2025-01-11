import pandas as pd
import pandas_ta as ta

data = pd.read_csv('../../project_raw_data/stock_data_apple_full_5min.csv')

# Berechne die Volatilität (Standardabweichung der Schlusskurse)
data['Volatility'] = data['Close'].rolling(window=14).std()

# Berechne den RSI
data['RSI'] = ta.rsi(data['Close'], length=14)

# Berechne den OBV
data['OBV'] = ta.obv(data['Close'], data['Volume'])

# Berechne den ATR
data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)

# Profitchange calculating and labelin
data['percent_change_close'] = data['Close'].pct_change()

data['Profit_Trend_Label'] = (data['percent_change_close'] > 0).astype(int)

data.drop(columns=['Close', 'percent_change_close', 'Dividends', 'Stock Splits'], inplace=True)

data.rename(columns={'Datetime':'Timestamp'}, inplace=True)

data = data[15:]

data.to_csv('stock_data_apple_indicators.csv', index=False)
