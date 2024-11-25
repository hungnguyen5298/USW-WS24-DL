import yfinance as yf
import pandas as pd

#Ticker
ticker = "AAPL"
apple_stock = yf.Ticker(ticker)

#Historische Kursdaten
historical_data = apple_stock.history(period="3mo")
info = apple_stock.info

df = pd.DataFrame(historical_data.copy())

df.to_csv("apple_stock_data.csv")