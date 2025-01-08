import pandas as pd

# Load the data
df = pd.read_csv('../data_preprocessing/data_segmenting/test.csv')
df2 = pd.read_csv('../project_raw_data/stock_data_apple_full_5min.csv')

# Get unique values of the time columns
time = df['timestamp_shifted']
time2 = df2['Datetime']

# Get the unique values of each time column
unique_time = time.unique()
unique_time2 = time2.unique()

# Find common time values
common_times = set(unique_time) & set(unique_time2)

# Output the results
print(f"Number of unique timestamps in df: {len(unique_time)}")
print(f"Number of unique timestamps in df2: {len(unique_time2)}")
print(f"Number of common timestamps: {len(common_times)}")
