import pandas as pd

vader = pd.read_csv('../data_preprocessing/data_segmenting/agg_sentiment_VADER.csv')
finbert = pd.read_csv('../data_preprocessing/data_segmenting/agg_sentiment_FinBERT.csv')

df = finbert.copy()
df1 = vader.copy()

# Filterung der Zeilen, bei denen alle drei Spaltenwerte 0.0 sind
zero_count = df[(df["Positive_Prob"] == 0.0) &
                (df["Neutral_Prob"] == 0.0) &
                (df["Negative_Prob"] == 0.0)].shape[0]

# Filterung der Zeilen, bei denen alle drei Spaltenwerte 0.0 sind
zero_count1 = df1[(df1["VADER_Positive"] == 0.0) &
                (df1["VADER_Neutral"] == 0.0) &
                (df1["VADER_Negative"] == 0.0)].shape[0]

print(f"Anzahl der Datensätze von FinBERT mit allen drei Werten = 0.0: {zero_count}")
print(f"Anzahl der Datensätze von VADER mit allen drei Werten = 0.0: {zero_count1}")


'''
Tracker von Weight.Loss werte,...
print(teacker)

Simbeck
'''