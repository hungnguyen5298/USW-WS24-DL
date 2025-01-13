import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Laden der Daten
data = pd.read_csv('../data_preprocessing/stock_data_feature_engineering/stock_data_apple_indicators_longer.csv')

data = data.drop(columns=['Timestamp'])
correlation = data.corr()
print(correlation['Change_Close'])

