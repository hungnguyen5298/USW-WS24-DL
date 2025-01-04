'''
Duplikate! can check lai. Tong chi co 168 Perioden
'''


import os

import pandas as pd
import numpy as np

# Đường dẫn tới file CSV chứa dữ liệu HLOCV
csv_file = "~/Dokumente/PycharmProjects/USW-WS24-DL/project_raw_data/stock_data_apple.csv"  # Đường dẫn file CSV
output_file = "~/Dokumente/PycharmProjects/USW-WS24-DL/data_preprocessing/stock_data_apple_indicators.csv"  # File xuất kết quả

# Hàm tính RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()  # Tính sự thay đổi giá
    gain = np.where(delta > 0, delta, 0)  # Chỉ lấy giá trị tăng
    loss = np.where(delta < 0, -delta, 0)  # Chỉ lấy giá trị giảm

    avg_gain = pd.Series(gain).rolling(window=window, min_periods=window).mean()
    avg_loss = pd.Series(loss).rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss  # Chỉ số sức mạnh tương đối
    rsi = 100 - (100 / (1 + rs))  # Tính RSI

    return pd.Series(rsi, index=data.index)


# Hàm tính MACD
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()

    macd = short_ema - long_ema  # Đường MACD
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()  # Đường tín hiệu

    return macd, signal_line

# Hàm tính EMA
def calculate_ema(data, span):
    return data['Close'].ewm(span=span, adjust=False).mean()

# Đọc dữ liệu từ file CSV
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"File {csv_file} không tồn tại. Hãy kiểm tra đường dẫn!")
    exit()

# Kiểm tra và xử lý nếu thiếu dữ liệu
if df.isnull().sum().sum() > 0:
    print("Dữ liệu bị thiếu. Đang xử lý các giá trị thiếu...")
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

# Tính toán các chỉ số kỹ thuật
print("Đang tính toán các chỉ số kỹ thuật...")
df['RSI'] = calculate_rsi(df)
df['MACD'], df['Signal_Line'] = calculate_macd(df)
df['EMA_20'] = calculate_ema(df, span=20)
df['EMA_50'] = calculate_ema(df, span=50)

# Lưu kết quả vào file CSV
print("Lưu kết quả vào file...")
os.makedirs(os.path.dirname(output_file), exist_ok=True)
df.to_csv(output_file, index=False, encoding="utf-8")

print(f"Chỉ số kỹ thuật đã được lưu vào {output_file}.")
