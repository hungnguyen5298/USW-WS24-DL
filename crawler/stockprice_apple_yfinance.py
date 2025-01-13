import os
import yfinance as yf
import pandas as pd

# Ticker
ticker = "AAPL"
apple_stock = yf.Ticker(ticker)

# Define start and end dates as keyword arguments
start_date = "2024-11-15"
end_date = "2025-06-01"

# Fetch historical data with a 30-minute interval
historical_data = apple_stock.history(start=start_date, end=end_date, interval="5m")
historical_data.reset_index(inplace=True)

# Format the 'Datetime' column
historical_data['Datetime'] = historical_data['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Path to save the CSV file
csv_file = "../project_raw_data/stock_data_apple_full_5m_longer.csv"

# Check if the CSV file already exists
if os.path.exists(csv_file):
    # Load existing data and combine with new data
    existing_df = pd.read_csv(csv_file)
    combined_df = pd.concat([existing_df, historical_data], ignore_index=True)
else:
    # Use only new data
    combined_df = historical_data

# Remove duplicates if necessary
combined_df = combined_df.drop_duplicates()

# Ensure the directory exists before saving
os.makedirs(os.path.dirname(csv_file), exist_ok=True)

# Save the combined data to the CSV file
combined_df.to_csv(csv_file, index=False, encoding="utf-8")

print(f"Data successfully saved to {csv_file}")
